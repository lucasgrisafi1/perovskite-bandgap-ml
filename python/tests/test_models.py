import numpy as np

from models import KRR_ALPHAS, KRR_GAMMA_MULTIPLIERS, KernelRidgeRBF, LinearRegressionNP, RandomForestNP, tune_krr


def test_linear_recovers_exact_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = 2 * X[:, 0] - X[:, 1] + 3
    m = LinearRegressionNP().fit(X, y)
    assert np.allclose(m.beta, [3, 2, -1])


def test_random_forest_learns_step_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 3))
    y = (X[:, 0] > 0).astype(float) * 5
    m = RandomForestNP(n_trees=20, min_leaf=3, max_features=1.0, seed=0).fit(X, y)
    assert np.mean((m.predict(X) - y) ** 2) < 0.1


def test_random_forest_max_features_fraction_and_seed_reproducible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 6))
    y = X[:, 0]
    p1 = RandomForestNP(n_trees=5, max_features=0.5, seed=3).fit(X, y).predict(X)
    p2 = RandomForestNP(n_trees=5, max_features=0.5, seed=3).fit(X, y).predict(X)
    assert np.array_equal(p1, p2)


def test_kernel_ridge_interpolates_smooth_function():
    X = np.linspace(0, 6, 60)[:, None]
    y = np.sin(X[:, 0])
    m = KernelRidgeRBF(alpha=1e-4, gamma=1.0).fit(X, y)
    assert np.sqrt(np.mean((m.predict(X) - y) ** 2)) < 0.05


def test_tune_krr_returns_grid_values():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] ** 2
    alpha, gamma, cv_rmse = tune_krr(X, y, n_folds=3)
    assert alpha in KRR_ALPHAS
    assert gamma in tuple(m / 3 for m in KRR_GAMMA_MULTIPLIERS)
    assert cv_rmse > 0


def test_tree_never_creates_empty_child_for_nearly_equal_values():
    # Adjacent values whose midpoint rounds to the larger one.
    from models import _Tree

    a = np.nextafter(1.0, 2.0)
    b = np.nextafter(a, 2.0)
    assert 0.5 * (a + b) == b  # the float trap this test guards against
    X = np.array([[a]] * 10 + [[b]] * 10)
    y = np.array([0.0] * 10 + [1.0] * 10)
    tree = _Tree(min_leaf=1, n_sub=1, rng=np.random.default_rng(0)).fit(X, y)
    assert np.all(np.isfinite(tree.value))
    assert np.allclose(tree.predict(X), y)
