"""Evaluation protocol: grouped K-fold CV and leave-one-element-out.

Model factories have the signature factory(X_train, y_train) -> fitted
model, so anything that must only see training data (hyperparameter
tuning) happens inside the factory. Standardization statistics are
always computed from the training rows of each split.
"""

import numpy as np


def compute_metrics(y_true, y_pred):
    """Return (R2, RMSE, MAE)."""
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot), float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def group_ids(sites):
    """Integer group per compound; key = (sorted A pair, sorted B pair).

    `sites` is a sequence of (a1, b1, a2, b2). Compounds with the same
    two A cations and the same two B cations (differing only in which A
    is paired with which B) have identical site features; 51 of the 87
    such pairs in the dataset are near-duplicates (gaps within 5 meV).
    Keeping each group on one side of every split prevents leakage.
    """
    keys = [(tuple(sorted((a1, a2))), tuple(sorted((b1, b2)))) for a1, b1, a2, b2 in sites]
    lookup = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([lookup[k] for k in keys])


def group_kfold(groups, k=5, seed=42):
    """Split row indices into k test folds; no group spans two folds.

    Groups are shuffled (seeded) and assigned greedily to the currently
    smallest fold so fold sizes stay balanced.
    """
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    folds, sizes = [[] for _ in range(k)], np.zeros(k, dtype=int)
    for g in uniq:
        f = int(np.argmin(sizes))
        folds[f].extend(members[g].tolist())
        sizes[f] += len(members[g])
    return [np.sort(np.array(f)) for f in folds]


def _standardize(X_tr, X_te):
    mu, sigma = X_tr.mean(0), X_tr.std(0, ddof=1)
    sigma[sigma == 0] = 1.0
    return (X_tr - mu) / sigma, (X_te - mu) / sigma


def cross_validate(factory, X, y, groups, k=5, seed=42, on_fold=None):
    """Grouped K-fold CV. Returns per-fold metrics, means/stds and OOF predictions.

    on_fold(model, X_test_scaled, y_test), if given, is called once per
    fold (e.g. to compute permutation importance on held-out data).
    """
    oof = np.empty(len(y))
    per_fold = []
    for te in group_kfold(groups, k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        X_tr, X_te = _standardize(X[tr], X[te])
        model = factory(X_tr, y[tr])
        oof[te] = model.predict(X_te)
        if on_fold is not None:
            on_fold(model, X_te, y[te])
        per_fold.append(compute_metrics(y[te], oof[te]))
    per_fold = np.array(per_fold)
    out = {"oof": oof}
    for j, name in enumerate(("r2", "rmse", "mae")):
        out[f"{name}_folds"] = per_fold[:, j]
        out[f"{name}_mean"] = float(per_fold[:, j].mean())
        out[f"{name}_std"] = float(per_fold[:, j].std(ddof=1))
    return out


def leave_element_out(factory, X, y, sites, elements=None):
    """For each element, train on compounds without it, test on those with it.

    Returns pooled metrics over all held-out predictions plus per-element
    RMSE and test-set sizes.
    """
    sites = list(sites)
    if elements is None:
        elements = sorted({e for s in sites for e in s})
    y_all, p_all, rmse, n_test = [], [], {}, {}
    for el in elements:
        te = np.array([el in s for s in sites])
        X_tr, X_te = _standardize(X[~te], X[te])
        pred = factory(X_tr, y[~te]).predict(X_te)
        y_all.append(y[te])
        p_all.append(pred)
        rmse[el] = float(np.sqrt(np.mean((y[te] - pred) ** 2)))
        n_test[el] = int(te.sum())
    y_all, p_all = np.concatenate(y_all), np.concatenate(p_all)
    r2, rmse_pooled, mae = compute_metrics(y_all, p_all)
    return {"r2": r2, "rmse": rmse_pooled, "mae": mae, "rmse_by_element": rmse, "n_test": n_test}


def permutation_importance(predict, X, y, column_groups=None, n_repeats=10, seed=42):
    """Mean increase in RMSE when a column (or group of columns) is shuffled.

    column_groups: list of column-index lists permuted together (rows are
    shuffled jointly, so within-group relationships are preserved). This
    gives a fair importance for correlated families such as the min/max/
    mean of one property. Defaults to one group per column.
    """
    if column_groups is None:
        column_groups = [[j] for j in range(X.shape[1])]
    rng = np.random.default_rng(seed)
    base = compute_metrics(y, predict(X))[1]
    imp = np.zeros(len(column_groups))
    for g, cols in enumerate(column_groups):
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            imp[g] += compute_metrics(y, predict(Xp))[1] - base
    return imp / n_repeats
