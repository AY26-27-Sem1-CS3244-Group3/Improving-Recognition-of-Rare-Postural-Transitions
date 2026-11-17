"""Fit and audit the LIME-style local probability surrogates for section 7.7."""
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, lasso_path
from sklearn.metrics import mean_absolute_error, r2_score
from threadpoolctl import threadpool_limits

from analyze import HERE, load_model

FIT_SUBJECTS = [26, 17, 25, 30, 21, 23, 14, 11, 16, 6, 1, 27, 8]
VALIDATION_SUBJECTS = [19, 29, 5, 7]
AUDIT_SUBJECTS = [28, 3, 22, 15]
ALPHA_FRACTIONS = np.geomspace(0.5, 0.001, 16)


def neighborhood(x, anchor, donors, radius, size, seed, sampler):
    rng = np.random.default_rng(seed)
    index = rng.integers(len(donors), size=size)
    fraction = rng.uniform(0, radius, size=size).astype(np.float32)
    displacement = (x[donors[index]] - anchor) * fraction[:, None]
    if sampler == "independent":
        for column in displacement.T:
            rng.shuffle(column)
    return np.ascontiguousarray(anchor + displacement, dtype=np.float32)


def kernel(z, anchor, bandwidth):
    return np.exp(-np.mean((z - anchor) ** 2, axis=1) / bandwidth**2).astype(float)


def anchored(z, y, w, anchor, anchor_y):
    return np.vstack([z, anchor]).astype(float), np.r_[y, anchor_y], np.r_[w, w.sum() * .05 / .95]


def path(z, y, w):
    center = np.average(z, axis=0, weights=w)
    mean = np.average(y, weights=w)
    scale = np.sqrt(w / w.mean())
    design = np.asfortranarray((z - center) * scale[:, None])
    response = (y - mean) * scale
    maximum = np.max(np.abs(design.T @ response)) / len(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        alphas, coef, gaps = lasso_path(
            design, response, alphas=maximum * ALPHA_FRACTIONS, max_iter=4000, tol=1e-5
        )
    intercept = mean - center @ coef
    return alphas, coef.T, intercept, gaps <= 1e-5 * np.mean(response**2)


def weighted_sd(y, w):
    return float(np.sqrt(np.average((y - np.average(y, weights=w))**2, weights=w)))


def run():
    model, train, _, _, x, test_x = load_model()
    torch.set_num_threads(1)
    pools = [np.flatnonzero(train.subject_id.isin(ids)) for ids in
             (FIT_SUBJECTS, VALIDATION_SUBJECTS, AUDIT_SUBJECTS)]
    cases = pd.read_csv(HERE / "surrogate_cases.csv")
    results, repeats = [], []
    for case in cases.itertuples(index=False):
        anchor = test_x[case.row]
        anchor_y = case.mlp_probability
        target = case.target - 1

        def predict(z):
            with torch.no_grad():
                return np.concatenate([
                    model(torch.from_numpy(z[i:i + 256])).softmax(1).numpy()[:, target]
                    for i in range(0, len(z), 256)
                ]).astype(float)

        np.testing.assert_allclose(predict(anchor[None, :]), anchor_y, atol=2e-6, rtol=1e-5)
        distance = np.mean((x - anchor)**2, axis=1)
        donors = [pool[np.argsort(distance[pool])[:256]] for pool in pools]
        larger_case = case.row == 51 and case.target == 6
        caps = (10, 20, 50) if larger_case else (10, 20)
        candidates = []
        for radius in (0.25, 0.5, 1.0):
            correlated = neighborhood(x, anchor, donors[0], radius, 768, 1101, "correlated")
            bandwidth = float(np.median(np.sqrt(np.mean((correlated - anchor)**2, axis=1))))
            for sampler in ("correlated", "independent"):
                fit = correlated if sampler == "correlated" else neighborhood(
                    x, anchor, donors[0], radius, 768, 1101, sampler
                )
                val = neighborhood(x, anchor, donors[1], radius, 384, 2202, sampler)
                val_y, val_w = predict(val), kernel(val, anchor, bandwidth)
                z, y, w = anchored(fit, predict(fit), kernel(fit, anchor, bandwidth), anchor, anchor_y)
                alphas, coefs, intercepts, converged = path(z, y, w)
                counts = (np.abs(coefs) > 1e-10).sum(axis=1)
                for cap in caps:
                    eligible = np.flatnonzero(converged & (counts > 0) & (counts <= cap))
                    if not len(eligible):
                        continue
                    j = max(eligible, key=lambda i: (counts[i], -alphas[i]))
                    columns = np.flatnonzero(np.abs(coefs[j]) > 1e-10)
                    ols = LinearRegression().fit(z[:, columns], y, sample_weight=w)
                    fits = [
                        ("lasso", coefs[j, columns], intercepts[j]),
                        ("post_lasso_ols", ols.coef_, ols.intercept_),
                    ]
                    for method, coef, intercept in fits:
                        estimate = val[:, columns] @ coef + intercept
                        candidates.append(dict(
                            cap=cap, sampler=sampler, radius=radius, method=method, columns=columns,
                            coef=coef, intercept=intercept, alpha=alphas[j], bandwidth=bandwidth,
                            validation_r2=r2_score(val_y, estimate, sample_weight=val_w),
                            validation_mae=mean_absolute_error(val_y, estimate, sample_weight=val_w),
                            validation_sd=weighted_sd(val_y, val_w),
                            anchor_error=abs(anchor[columns] @ coef + intercept - anchor_y),
                        ))

        sparse = [a for a in candidates if a["cap"] <= 20]
        informative = [a for a in sparse if a["validation_sd"] >= .02]
        accurate = [a for a in informative if a["validation_mae"] <= .05 and a["anchor_error"] <= .05]
        choices = {"sparse": max(accurate or informative or sparse,
                                 key=lambda a: (a["validation_r2"], -len(a["columns"])))}
        if larger_case:
            choices["larger"] = max(
                (a for a in candidates if a["cap"] == 50 and a["validation_sd"] >= .02),
                key=lambda a: a["validation_r2"],
            )

        for fit_type, chosen in choices.items():
            sampler, radius, bandwidth = (chosen[key] for key in ("sampler", "radius", "bandwidth"))
            columns = chosen["columns"]
            audit = neighborhood(x, anchor, donors[2], radius, 768, 3303, sampler)
            audit_y, audit_w = predict(audit), kernel(audit, anchor, bandwidth)
            estimate = audit[:, columns] @ chosen["coef"] + chosen["intercept"]
            r2 = r2_score(audit_y, estimate, sample_weight=audit_w)
            mae = mean_absolute_error(audit_y, estimate, sample_weight=audit_w)
            sd = weighted_sd(audit_y, audit_w)
            accepted = len(columns) <= 20 and r2 >= .8 and mae <= .05 and sd >= .02 and chosen["anchor_error"] <= .05
            results.append(dict(row=case.row, target=case.target, sampler=sampler, radius=radius,
                                method=chosen["method"], nonzero=len(columns), validation_r2=chosen["validation_r2"],
                                audit_r2=r2, audit_mae=mae, audit_sd=sd, anchor_error=chosen["anchor_error"],
                                passes=accepted, fit=fit_type))

            if fit_type == "larger":
                assert chosen["method"] == "post_lasso_ols"
                support = set(columns)
                for seed in (601, 602, 603):
                    fit = neighborhood(x, anchor, donors[0], radius, 768, seed, sampler)
                    z, y, w = anchored(fit, predict(fit), kernel(fit, anchor, bandwidth), anchor, anchor_y)
                    lasso = Lasso(alpha=chosen["alpha"], max_iter=30000, tol=1e-5).fit(z, y, sample_weight=w)
                    selected = np.flatnonzero(np.abs(lasso.coef_) > 1e-10)
                    ols = LinearRegression().fit(z[:, selected], y, sample_weight=w)
                    other = set(selected)
                    left, right = np.intersect1d(columns, selected, return_indices=True)[1:]
                    sign = np.mean(np.sign(chosen["coef"][left]) == np.sign(ols.coef_[right]))
                    repeats.append(dict(seed=seed, nonzero=len(selected),
                                        audit_r2=r2_score(audit_y, ols.predict(audit[:, selected]), sample_weight=audit_w),
                                        jaccard=len(support & other) / len(support | other), sign_agreement=sign))
        print(f"Audited row {case.row}, class {case.target}", flush=True)
    pd.DataFrame(results).to_csv(HERE / "surrogate_results.csv", index=False)
    pd.DataFrame(repeats).to_csv(HERE / "surrogate_refits.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
