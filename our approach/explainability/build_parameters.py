"""Build the fixed seven-window LDA parameters used by evaluation section 7.7."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).with_name("lda_fixed_parameters.npz")


def source_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def recorded_predictions(path):
    notebook = json.loads(path.read_bytes())
    matches = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "recorded_selected_predictions"
                for target in node.targets
            ):
                matches.append(node.value)
    if len(matches) != 1:
        raise ValueError("Expected one recorded_selected_predictions assignment in the companion.")
    value = matches[0]
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "np"
        and value.func.attr == "array"
        and len(value.args) == 1
    ):
        raise ValueError("Expected a saved literal np.array of test predictions.")
    predictions = np.asarray(ast.literal_eval(value.args[0]))
    if predictions.ndim != 1 or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Recorded predictions must be a one-dimensional integer array.")
    if not np.isin(predictions, np.arange(1, 13)).all():
        raise ValueError("Recorded predictions contain an unknown class.")
    return predictions


def add_context(values, groups):
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(groups) - 1]
    first = np.repeat(starts, ends - starts + 1)
    last = np.repeat(ends, ends - starts + 1)
    row = np.arange(len(values))
    return np.hstack([
        values[np.clip(row + offset, first, last)] for offset in range(-3, 4)
    ])


def posture_memory(probabilities, groups):
    prediction = probabilities.argmax(axis=1)
    start = 0
    for row in range(len(prediction)):
        if row == 0 or groups[row] != groups[row - 1]:
            start = row
        if prediction[row] not in (8, 10):
            continue
        mass = probabilities[row, [8, 10]].sum()
        score = probabilities[row, 10] / mass if mass else 0.5
        past = probabilities[max(start, row - 8):row, [3, 4]].sum(axis=0)
        if past.sum() >= 0.75:
            score = past[1] / past.sum()
        prediction[row] = 10 if score > 0.5 else 8
    return prediction + 1


def build_parameters(output):
    import pandas as pd
    import sklearn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    source_dir = ROOT / "our approach"
    train_path = source_dir / "wrangled data/hapt_train.csv"
    test_path = source_dir / "wrangled data/hapt_test.csv"
    main_path = source_dir / "Improving Rare Postural Transitions.ipynb"
    companion_path = source_dir / "companion/LDA external.ipynb"
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    features = [c for c in train.columns if c not in ["subject_id", "activity_id", "activity_label"]]
    if len(features) != 561 or list(test.columns) != list(train.columns):
        raise ValueError("Expected matching train/test columns with 561 features.")
    train_groups = train.subject_id.to_numpy(dtype=np.int64)
    test_groups = test.subject_id.to_numpy(dtype=np.int64)
    if set(train_groups) & set(test_groups):
        raise ValueError("Training and test participants overlap.")
    recorded = recorded_predictions(companion_path)
    if recorded.shape != (len(test),):
        raise ValueError("Recorded predictions do not match the test row count.")
    train_context = add_context(train[features].to_numpy(dtype=np.float32), train_groups)
    test_context = add_context(test[features].to_numpy(dtype=np.float32), test_groups)

    print("Fitting the selected seven-window LDA once; no CV or tuning.", flush=True)
    with threadpool_limits(limits=2):
        started = perf_counter()
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(train_context).astype(np.float32)
        eval_x = scaler.transform(test_context).astype(np.float32)
        model = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=np.full(12, 1 / 12)
        )
        model.fit(fit_x, train.activity_id.to_numpy(dtype=np.int64))
        np.testing.assert_array_equal(model.classes_, np.arange(1, 13))
        probabilities = model.predict_proba(eval_x)
        predictions = posture_memory(probabilities, test_groups)
        elapsed = perf_counter() - started

    mismatch = np.flatnonzero(predictions != recorded)
    if len(mismatch):
        raise ValueError(
            f"Refit differs from {len(mismatch)} recorded labels; first rows: "
            f"{mismatch[:10].tolist()}. No parameters were saved."
        )
    scores = eval_x @ model.coef_.T + model.intercept_
    exponent = np.exp(scores - scores.max(axis=1, keepdims=True))
    np.testing.assert_allclose(
        exponent / exponent.sum(axis=1, keepdims=True), probabilities, rtol=1e-10, atol=1e-12
    )
    metadata = {
        "purpose": "Single fixed-configuration fit for local explainability",
        "model_fit_count": 1,
        "cv_or_search_performed": False,
        "configuration": "7-window raw StandardScaler + LDA(lsqr, shrinkage=auto, equal priors) + posture memory",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "fit_and_prediction_seconds": elapsed,
        "recorded_prediction_mismatches": 0,
        "training_csv_sha256": source_hash(train_path),
        "test_csv_sha256": source_hash(test_path),
        "main_sha256": source_hash(main_path),
        "companion_sha256": source_hash(companion_path),
        "builder_sha256": source_hash(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        np.savez_compressed(
            destination,
            coefficients=model.coef_,
            intercept=model.intercept_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            feature_names=np.asarray(features),
            classes=model.classes_,
            reference_scaled=fit_x.mean(axis=0),
            test_probabilities=probabilities,
            final_predictions=predictions,
            metadata=np.asarray(json.dumps(metadata)),
        )
    print(f"Saved {output}; all {len(recorded):,} labels match the recorded results.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", help="Fit the fixed LDA once and save parameters.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New .npz path; existing files are never overwritten.")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not args.fit:
        print("Training disabled. Add --fit and choose a new --output path to build parameters.")
        print(f"Parameter file: {output} ({'present' if output.is_file() else 'absent'})")
        return 0
    if output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz; no training was performed.")
    if output.exists():
        parser.error("Output already exists. Choose a new --output path; no training was performed.")
    build_parameters(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
