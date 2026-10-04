"""Regression models implemented from scratch in numpy (no scikit-learn).

LinearRegressionNP  ordinary least squares with intercept
RandomForestNP      bagged CART regression trees with random feature subsets
KernelRidgeRBF      kernel ridge regression with a Gaussian (RBF) kernel;
                    same kernel family as MATLAB's fitrsvm RBF, but a squared
                    loss instead of the SVM's epsilon-insensitive loss
tune_krr            K-fold grid search of KRR (alpha, gamma)
"""

import numpy as np


class LinearRegressionNP:
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        self.beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.beta


class _Tree:
    """CART regression tree (variance reduction) with random feature subsets."""

    def __init__(self, min_leaf, n_sub, rng):
        self.min_leaf, self.n_sub, self.rng = min_leaf, n_sub, rng

    def fit(self, X, y):
        self.feat, self.thresh, self.left, self.right, self.value = [], [], [], [], []
        self._grow(X, y)
        self.feat = np.array(self.feat)
        self.thresh = np.array(self.thresh)
        self.left = np.array(self.left)
        self.right = np.array(self.right)
        self.value = np.array(self.value)
        return self

    def _new_node(self, value):
        for lst, v in ((self.feat, -1), (self.thresh, 0.0), (self.left, -1), (self.right, -1), (self.value, value)):
            lst.append(v)
        return len(self.value) - 1

    def _grow(self, X, y):
        idx = self._new_node(y.mean())
        if len(y) >= 2 * self.min_leaf and np.ptp(y) > 0:
            best = self._best_split(X, y)
            if best is not None:
                f, t = best
                mask = X[:, f] <= t
                self.feat[idx], self.thresh[idx] = f, t
                self.left[idx] = self._grow(X[mask], y[mask])
                self.right[idx] = self._grow(X[~mask], y[~mask])
        return idx

    def _best_split(self, X, y):
        n, d = X.shape
        m = self.min_leaf
        feats = self.rng.choice(d, size=min(self.n_sub, d), replace=False)
        parent = np.var(y) * n
        best, best_gain = None, 1e-12
        nl = np.arange(m, n - m + 1)          # left-child sizes allowed by min_leaf
        nr = n - nl
        for f in feats:
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            csum, csq = np.cumsum(ys), np.cumsum(ys ** 2)
            i = nl - 1                         # last index of the left child
            sse_l = csq[i] - csum[i] ** 2 / nl
            sse_r = (csq[-1] - csq[i]) - (csum[-1] - csum[i]) ** 2 / nr
            gain = parent - (sse_l + sse_r)
            gain[xs[i] == xs[i + 1]] = -np.inf  # cannot split between equal values
            j = int(np.argmax(gain))
            if gain[j] > best_gain:
                best_gain = gain[j]
                lo, hi = xs[i[j]], xs[i[j] + 1]
                t = 0.5 * (lo + hi)
                # For adjacent floats the midpoint can round up to `hi`,
                # which would send every row left and leave an empty child.
                best = (f, t if t < hi else lo)
        return best

    def predict(self, X):
        node = np.zeros(len(X), dtype=int)
        active = self.feat[node] >= 0
        while active.any():
            n = node[active]
            go_left = X[active, self.feat[n]] <= self.thresh[n]
            node[active] = np.where(go_left, self.left[n], self.right[n])
            active = self.feat[node] >= 0
        return self.value[node]


class RandomForestNP:
    """Random forest regressor.

    max_features: fraction of features (float in (0, 1]) or count (int)
    tried at each split. Default 1/3, the usual regression choice.
    """

    def __init__(self, n_trees=100, min_leaf=5, max_features=1 / 3, seed=42):
        self.n_trees, self.min_leaf, self.max_features, self.seed = n_trees, min_leaf, max_features, seed

    def _n_sub(self, d):
        if isinstance(self.max_features, (int, np.integer)):
            return max(1, min(d, int(self.max_features)))
        return max(1, int(np.ceil(self.max_features * d)))

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        n_sub = self._n_sub(d)
        self.trees = []
        for _ in range(self.n_trees):
            boot = rng.integers(0, n, size=n)
            self.trees.append(_Tree(self.min_leaf, n_sub, rng).fit(X[boot], y[boot]))
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees], axis=0)


class KernelRidgeRBF:
    def __init__(self, alpha=1.0, gamma=None):
        self.alpha, self.gamma = alpha, gamma

    def _kernel(self, A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-self.gamma * np.maximum(d2, 0.0))

    def fit(self, X, y):
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        self.X = X
        K = self._kernel(X, X)
        self.dual = np.linalg.solve(K + self.alpha * np.eye(len(X)), y)
        return self

    def predict(self, X):
        return self._kernel(X, self.X) @ self.dual


KRR_ALPHAS = (0.01, 0.1, 1.0, 10.0)
KRR_GAMMA_MULTIPLIERS = (0.25, 1.0, 4.0)  # times 1 / n_features


def tune_krr(X, y, n_folds=5, seed=42):
    """Grid-search KRR (alpha, gamma) by K-fold CV on (X, y) only.

    Returns (alpha, gamma, mean CV RMSE). Call it on training data only.
    """
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), n_folds)
    d = X.shape[1]
    best = (1.0, 1.0 / d, np.inf)
    for alpha in KRR_ALPHAS:
        for mult in KRR_GAMMA_MULTIPLIERS:
            gamma = mult / d
            rmses = []
            for k in range(n_folds):
                va = folds[k]
                tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                pred = KernelRidgeRBF(alpha, gamma).fit(X[tr], y[tr]).predict(X[va])
                rmses.append(np.sqrt(np.mean((y[va] - pred) ** 2)))
            if np.mean(rmses) < best[2]:
                best = (alpha, gamma, float(np.mean(rmses)))
    return best
