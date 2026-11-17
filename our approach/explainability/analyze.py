"""Window occlusion and Integrated Gradients for the fixed 11-window MLP."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch import nn

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "wrangled data"
COMPANION = HERE.parent / "companion" / "MLP external.ipynb"
CHECKPOINT = HERE / "mlp_fixed_refit.pt"
OFFSETS = np.arange(-5, 6)
WIDTH = 561


def add_context(frame, feature_columns):
    values = frame[feature_columns].to_numpy(dtype=np.float32)
    groups = frame.subject_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(groups) - 1]
    first = np.repeat(starts, ends - starts + 1)
    last = np.repeat(ends, ends - starts + 1)
    row = np.arange(len(values))
    return np.hstack([values[np.clip(row + offset, first, last)] for offset in range(-5, 6)])



def load_model():
    torch.set_num_threads(2)
    train = pd.read_csv(DATA / "hapt_train.csv")
    test = pd.read_csv(DATA / "hapt_test.csv")
    columns = train.columns.drop(["subject_id", "activity_id", "activity_label"]).tolist()
    assert len(columns) == WIDTH and test.columns.equals(train.columns)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(add_context(train, columns))
    test_x = scaler.transform(add_context(test, columns))
    assert train_x.shape == (7767, 6171) and test_x.shape == (3162, 6171)
    model = nn.Sequential(
        nn.Linear(6171,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,12))
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    return model.eval(), train, test, columns, train_x, test_x


def recorded_predictions():
    cells = json.loads(COMPANION.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "recorded_selected_predictions = " not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "recorded_selected_predictions" for t in node.targets
            ):
                return np.fromstring(ast.literal_eval(node.value.args[0]), sep=" ", dtype=int)
    raise ValueError("MLP companion predictions are missing.")


def logits(model, values, batch_size=512):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(values[i:i+batch_size].astype(np.float32))).numpy()
                               for i in range(0,len(values),batch_size)])

def posture_memory(probabilities, groups):
    pred = probabilities.argmax(axis=1)
    start = 0
    for row in range(len(pred)):
        if row == 0 or groups[row] != groups[row-1]:
            start = row
        if pred[row] not in (8,10):
            continue
        mass = probabilities[row,[8,10]].sum()
        score = probabilities[row,10]/mass if mass else 0.5
        past = probabilities[max(start,row-8):row,[3,4]].sum(axis=0)
        if past.sum() >= 0.75:
            score = past[1]/past.sum()
        pred[row] = 10 if score > 0.5 else 8
    return pred + 1

def integrated_gradients(model, values, ref, target_index, competitor_index, intervals=64, batch=4):
    result = np.empty_like(values,dtype=np.float32)
    alpha = torch.linspace(0,1,intervals+1).view(intervals+1,1,1)
    trap = torch.ones(intervals+1); trap[0]=trap[-1]=.5
    for start in range(0,len(values),batch):
        stop = min(start+batch,len(values))
        x = torch.from_numpy(values[start:stop])
        base = torch.from_numpy(ref).view(1,1,-1)
        path = (base+alpha*(x.unsqueeze(0)-base)).detach().requires_grad_(True)
        flat = path.reshape(-1,values.shape[1])
        out = model(flat)
        t = torch.from_numpy(np.tile(target_index[start:stop],intervals+1))
        c = torch.from_numpy(np.tile(competitor_index[start:stop],intervals+1))
        score = (out[torch.arange(len(out)),t]-out[torch.arange(len(out)),c]).sum()
        grad = torch.autograd.grad(score,path)[0]
        mean_grad = (grad*trap.view(-1,1,1)).sum(dim=0)/intervals
        result[start:stop] = ((x-torch.from_numpy(ref))*mean_grad).detach().numpy()
    return result


def analyze():
    model, train, test, columns, train_x, test_x = load_model()
    test_logits = logits(model, test_x)
    probabilities = torch.from_numpy(test_logits).softmax(1).numpy()
    final = posture_memory(probabilities, test.subject_id.to_numpy())
    np.testing.assert_array_equal(final, recorded_predictions())

    rows = np.flatnonzero(test.activity_id.to_numpy() >= 7)
    actual = test.activity_id.to_numpy()[rows]
    x = test_x[rows]
    order = np.argsort(test_logits[rows], axis=1)
    target = order[:, -1]
    comparator = np.where(target + 1 == actual, order[:, -2], actual - 1)
    ix = np.arange(len(rows))
    margin = test_logits[rows, target] - test_logits[rows, comparator]
    np.testing.assert_array_equal(test_logits[rows].argmax(1), target)
    labels = train.set_index("activity_id").activity_label.to_dict()
    references = {"Training mean": np.zeros(x.shape[1], dtype=x.dtype)}
    reference_rows = {}
    for class_id in (4, 5, 6):
        ids = np.flatnonzero(train.activity_id.to_numpy() == class_id)
        region = train_x[ids]
        i = ids[np.argmin(np.sum((region - region.mean(axis=0)) ** 2, axis=1))]
        name = labels[class_id].capitalize()
        references[name] = train_x[i]
        reference_rows[name] = int(i)

    by_offset, by_group, checks, features = [], [], [], []
    examples = {65: int(np.flatnonzero(rows == 65)[0]), 51: int(np.flatnonzero(rows == 51)[0])}
    previous = (OFFSETS.tolist().index(-1)) * WIDTH
    start, stop = previous, previous + WIDTH
    row65_features = ["tBodyAcc-Mean-1", "tGravityAcc-Mad-1", "tGravityAcc-Min-1"]
    for name, reference in references.items():
        drops = []
        for j in range(len(OFFSETS)):
            block = slice(j * WIDTH, (j + 1) * WIDTH)
            altered = x.copy()
            altered[:, block] = reference[block]
            out = logits(model, altered)
            drop = margin - (out[ix, target] - out[ix, comparator])
            drops.append(drop)
            if OFFSETS[j] == -1:
                case65_top = int(out[examples[65]].argmax() + 1)
                assert case65_top == actual[examples[65]]
            by_offset.append((name, OFFSETS[j], np.abs(drop).mean()))

        per_class = pd.DataFrame(np.abs(drops).T).groupby(actual).mean()
        assert (per_class.idxmax(axis=1) == 5).all()
        groups = {"Past": OFFSETS < 0, "Future": OFFSETS > 0, "All neighbours": OFFSETS != 0}
        for group, mask in groups.items():
            selected = np.repeat(mask, WIDTH)
            altered = x.copy()
            altered[:, selected] = reference[selected]
            out = logits(model, altered)
            drop = margin - (out[ix, target] - out[ix, comparator])
            by_group.append((name, group, len(rows), int((drop > 0).sum()), drop.mean()))

        attr = integrated_gradients(model, x, reference, target, comparator)
        ref_score = logits(model, reference[None, :])[0]
        change = margin - (ref_score[target] - ref_score[comparator])
        residual = attr.sum(axis=1) - change
        refine = (np.abs(residual) > 0.1) | (np.abs(residual) > 0.05 * np.maximum(np.abs(change), 1e-6))
        if refine.any():
            attr[refine] = integrated_gradients(
                model, x[refine], reference, target[refine], comparator[refine], intervals=256, batch=2
            )
        residual = np.abs(attr.sum(axis=1) - change)
        signed_by_window = attr.reshape(len(rows), len(OFFSETS), WIDTH).sum(axis=-1)
        signed_drops = np.array(drops).T
        neighbours = OFFSETS != 0
        rho = [
            spearmanr(a[neighbours], b[neighbours]).statistic
            for a, b in zip(signed_drops, signed_by_window)
        ]
        checks.append((name, int(refine.sum()), np.median(residual), np.quantile(residual, .95), np.median(rho)))

        if name == "Training mean":
            ranked = np.argsort(-np.abs(attr[examples[51], start:stop]))[:5]
            row51_indices = start + ranked
        for row, k in examples.items():
            block = attr[k, start:stop]
            if row == 65:
                selections = [(feature, [start + columns.index(feature)]) for feature in row65_features]
            else:
                selections = [(", ".join(columns[j - start] for j in row51_indices), row51_indices)]
            for description, indices in selections:
                altered = x[k].copy()
                altered[indices] = reference[indices]
                score = logits(model, altered[None, :])[0]
                after = score[target[k]] - score[comparator[k]]
                rank = 1 + int((np.abs(block) > abs(attr[k, indices[0]])).sum()) if len(indices) == 1 else None
                features.append((
                    row, -1, name, description, attr[k, indices].sum(), rank,
                    margin[k] - after, labels[score.argmax() + 1],
                ))
    summary = {
        "matched_final_predictions": len(final),
        "transition_windows": len(rows),
        "participants": int(test.iloc[rows].subject_id.nunique()),
        "base_transition_correct": int((target + 1 == actual).sum()),
        "final_transition_correct": int((final[rows] == actual).sum()),
        "reference_training_rows": reference_rows,
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    }
    (HERE / "model_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    for filename, records, headers in [
        ("window_effects.csv", by_offset, ["reference", "offset", "mean_absolute_drop"]),
        ("joint_effects.csv", by_group, ["reference", "group", "windows", "margin_decreased", "mean_drop"]),
        ("ig_checks.csv", checks, ["reference", "refined", "median_residual", "p95_residual", "neighbour_spearman"]),
        ("feature_cases.csv", features, ["row", "offset", "reference", "features", "signed_ig",
                                        "absolute_ig_rank", "margin_drop", "ablated_prediction"]),
    ]:
        pd.DataFrame(records, columns=headers).to_csv(HERE / filename, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    analyze()
