"""Materials Informatics: Perovskite Band Gap Predictor (Python mirror).

Predicts GLLB-SC band gaps of 1306 double perovskites (Pilania et al.,
Sci. Rep. 2016; matminer 'double_perovskites_gap') from 5 compositional
descriptors. Pure numpy implementations (no sklearn): OLS linear
regression, random forest, and RBF kernel ridge regression.

The MATLAB pipeline in ../matlab is the primary implementation
(fitlm / TreeBagger / fitrsvm); this script mirrors it for verification.

Run:  python3 bandgap_pipeline.py
"""

import re
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from element_data import ELEMENT_PROPS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "double_perovskites_gap.csv")
OUT = os.path.join(HERE, "..", "output")

FEATURE_NAMES = ["Mean EN", "Std EN", "Mean Radius", "Mean Valence", "Radius Ratio"]


# ── Formula parsing ──────────────────────────────────────────────────────────
def parse_formula(formula):
    """'AgNbLaAlO6' -> (['Ag','Nb','La','Al','O'], [1,1,1,1,6])"""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    elements, counts = [], []
    for sym, num in tokens:
        if not sym:
            continue
        elements.append(sym)
        counts.append(int(num) if num else 1)
    return elements, counts


# ── Feature engineering ──────────────────────────────────────────────────────
def featurize(formula):
    """5 compositional descriptors capturing orbital-overlap physics."""
    elements, counts = parse_formula(formula)
    # Validate: regex silently drops anything it can't match, and an
    # unknown element would raise a bare KeyError deep in the lookup.
    consumed = "".join(
        e + (str(c) if c > 1 else "") for e, c in zip(elements, counts)
    )
    if consumed != formula:
        raise ValueError(f"Could not fully parse formula: {formula!r}")
    unknown = [e for e in elements if e not in ELEMENT_PROPS]
    if unknown:
        raise ValueError(f"No element data for {unknown} in {formula!r}")
    props = np.array([ELEMENT_PROPS[e][1:] for e in elements], dtype=float)
    en, radius, valence = props[:, 0], props[:, 1], props[:, 2]
    w = np.array(counts, dtype=float)
    w = w / w.sum()
    return np.array([
        np.sum(w * en),                  # mean electronegativity
        np.std(en),                      # EN diversity (unweighted, per plan)
        np.sum(w * radius),              # mean atomic radius
        np.sum(w * valence),             # mean valence electrons
        radius.max() / radius.min(),     # radius ratio (stability proxy)
    ])


# ── Models (numpy-only) ──────────────────────────────────────────────────────
class LinearRegressionNP:
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        self.beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.beta


class _Tree:
    """CART regression tree (variance reduction), random feature subsets."""

    def __init__(self, min_leaf, n_sub, rng):
        self.min_leaf, self.n_sub, self.rng = min_leaf, n_sub, rng

    def fit(self, X, y):
        self.nodes = []
        self._grow(X, y)
        return self

    def _grow(self, X, y):
        node = {"value": y.mean()}
        idx = len(self.nodes)
        self.nodes.append(node)
        if len(y) >= 2 * self.min_leaf and np.ptp(y) > 0:
            best = self._best_split(X, y)
            if best is not None:
                f, t = best
                mask = X[:, f] <= t
                node["feat"], node["thresh"] = f, t
                node["left"] = self._grow(X[mask], y[mask])
                node["right"] = self._grow(X[~mask], y[~mask])
        return idx

    def _best_split(self, X, y):
        n, d = X.shape
        feats = self.rng.choice(d, size=min(self.n_sub, d), replace=False)
        parent = np.var(y) * n
        best, best_gain = None, 1e-12
        for f in feats:
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            csum, csq = np.cumsum(ys), np.cumsum(ys**2)
            tot, totsq = csum[-1], csq[-1]
            for i in range(self.min_leaf - 1, n - self.min_leaf):
                if xs[i] == xs[i + 1]:
                    continue
                nl = i + 1
                nr = n - nl
                sse_l = csq[i] - csum[i] ** 2 / nl
                sse_r = (totsq - csq[i]) - (tot - csum[i]) ** 2 / nr
                gain = parent - (sse_l + sse_r)
                if gain > best_gain:
                    best_gain = gain
                    best = (f, 0.5 * (xs[i] + xs[i + 1]))
        return best

    def predict(self, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            node = self.nodes[0]
            while "feat" in node:
                node = self.nodes[node["left"] if x[node["feat"]] <= node["thresh"] else node["right"]]
            out[i] = node["value"]
        return out


class RandomForestNP:
    def __init__(self, n_trees=100, min_leaf=5, seed=42):
        self.n_trees, self.min_leaf, self.seed = n_trees, min_leaf, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        n_sub = max(1, int(np.ceil(d / 3)))
        self.trees = []
        for _ in range(self.n_trees):
            boot = rng.integers(0, n, size=n)
            self.trees.append(_Tree(self.min_leaf, n_sub, rng).fit(X[boot], y[boot]))
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees], axis=0)


class KernelRidgeRBF:
    """RBF kernel stand-in for MATLAB's fitrsvm (same kernel family)."""

    def __init__(self, alpha=1.0, gamma=None):
        self.alpha, self.gamma = alpha, gamma

    def _kernel(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-self.gamma * d2)

    def fit(self, X, y):
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        self.X = X
        K = self._kernel(X, X)
        self.dual = np.linalg.solve(K + self.alpha * np.eye(len(X)), y)
        return self

    def predict(self, X):
        return self._kernel(X, self.X) @ self.dual


# ── Metrics & importance ─────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot, np.sqrt(np.mean((y_true - y_pred) ** 2))


def permutation_importance(model, X, y, seed=42, n_repeats=10):
    rng = np.random.default_rng(seed)
    base = compute_metrics(y, model.predict(X))[1]
    imp = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, f] = rng.permutation(Xp[:, f])
            imp[f] += compute_metrics(y, model.predict(Xp))[1] - base
    return imp / n_repeats


def tune_krr(X, y, n_folds=5, seed=42):
    """Grid-search KRR (alpha, gamma) by K-fold CV on the training set."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, n_folds)
    d = X.shape[1]
    best = (1.0, 1.0 / d, np.inf)
    for alpha in (0.01, 0.1, 1.0, 10.0):
        for gamma in (0.25 / d, 1.0 / d, 4.0 / d):
            rmses = []
            for k in range(n_folds):
                va = folds[k]
                trn = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                m = KernelRidgeRBF(alpha=alpha, gamma=gamma).fit(X[trn], y[trn])
                rmses.append(compute_metrics(y[va], m.predict(X[va]))[1])
            mean_rmse = float(np.mean(rmses))
            if mean_rmse < best[2]:
                best = (alpha, gamma, mean_rmse)
    return best


# ── Pipeline ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)

    # Load + filter (tolerate blank/missing gap values instead of crashing)
    with open(DATA) as f:
        rows = list(csv.DictReader(f))
    rows = [
        r for r in rows
        if r.get("gap_gllbsc", "").strip() and float(r["gap_gllbsc"]) > 0
    ]
    print(f"Compounds after filtering (gap > 0): {len(rows)}")

    X = np.array([featurize(r["formula"]) for r in rows])
    y = np.array([float(r["gap_gllbsc"]) for r in rows])

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid], y[valid]
    print(f"Final dataset: {len(y)} compounds, {X.shape[1]} features, no NaNs")

    # Split 80/20 FIRST, then normalize with train-only statistics.
    # (Normalizing before the split leaks test-set statistics into
    # training — a small but real form of data leakage.)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    n_test = int(round(0.2 * len(y)))
    te, tr = perm[:n_test], perm[n_test:]
    mu, sigma = X[tr].mean(0), X[tr].std(0, ddof=1)
    sigma[sigma == 0] = 1.0                      # guard constant features
    X_tr, y_tr = (X[tr] - mu) / sigma, y[tr]
    X_te, y_te = (X[te] - mu) / sigma, y[te]
    print(f"Train: {len(tr)}  Test: {len(te)}")

    # Tune KRR hyperparameters by 5-fold CV on the training set only
    alpha, gamma, cv_rmse = tune_krr(X_tr, y_tr)
    print(f"KRR tuned: alpha={alpha:g}, gamma={gamma:g} (CV RMSE={cv_rmse:.3f})")

    # Train 3 models
    models = {
        "Linear Regression": LinearRegressionNP().fit(X_tr, y_tr),
        "Random Forest": RandomForestNP(100, 5, seed=42).fit(X_tr, y_tr),
        "Kernel Ridge (RBF)": KernelRidgeRBF(alpha=alpha, gamma=gamma).fit(X_tr, y_tr),
    }

    results, preds = {}, {}
    print(f"\n{'Model':<20}{'R2':>8}{'RMSE':>8}")
    for name, m in models.items():
        yhat = m.predict(X_te)
        preds[name] = yhat
        r2, rmse = compute_metrics(y_te, yhat)
        results[name] = (r2, rmse)
        print(f"{name:<20}{r2:>8.3f}{rmse:>8.3f}")

    # Figure 1: predicted vs actual
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    colors = ["#3380cc", "#1a994d", "#cc4d33"]
    for ax, (name, yhat), c in zip(axes, preds.items(), colors):
        ax.scatter(y_te, yhat, s=14, c=c, alpha=0.6, edgecolors="none")
        lims = [min(y_te) * 0.9, max(y_te) * 1.1]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims), ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("Actual Band Gap (eV)")
        ax.set_ylabel("Predicted Band Gap (eV)")
        ax.set_title(f"{name}\nR$^2$={results[name][0]:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "predicted_vs_actual_python.png"), dpi=300)

    # Figure 2: RF permutation importance
    imp = permutation_importance(models["Random Forest"], X_te, y_te)
    order = np.argsort(imp)
    fig2, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.barh([FEATURE_NAMES[i] for i in order], imp[order], color="#339966")
    ax.set_xlabel("Increase in test RMSE when permuted (eV)")
    ax.set_title("Feature Importance — Random Forest", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "feature_importance_python.png"), dpi=300)

    # Save comparison table
    with open(os.path.join(OUT, "model_comparison_python.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "R2", "RMSE_eV"])
        for name, (r2, rmse) in results.items():
            w.writerow([name, f"{r2:.4f}", f"{rmse:.4f}"])
    print("\nSaved figures + model_comparison_python.csv to output/")

    imp_rank = sorted(zip(FEATURE_NAMES, imp), key=lambda t: -t[1])
    print("Feature importance ranking:", [f"{n}: {v:.3f}" for n, v in imp_rank])


if __name__ == "__main__":
    main()
