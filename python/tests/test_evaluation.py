import numpy as np

from evaluation import compute_metrics, cross_validate, group_ids, group_kfold, leave_element_out
from models import LinearRegressionNP


def test_metrics_perfect_and_known():
    assert compute_metrics(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == (1.0, 0.0, 0.0)
    r2, rmse, mae = compute_metrics(np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert (r2, rmse, mae) == (0.0, 1.0, 1.0)


def test_group_ids_merge_site_relabelled_duplicates():
    g = group_ids([("Ag", "Ta", "Cs", "Nb"), ("Cs", "Ta", "Ag", "Nb"),
                   ("Ag", "Nb", "Cs", "Ta"), ("Ba", "Ti", "Sr", "Ti")])
    assert g[0] == g[1] == g[2] != g[3]


def test_group_kfold_partitions_without_splitting_groups():
    groups = np.repeat(np.arange(50), 3)
    folds = group_kfold(groups, k=5, seed=0)
    assert len(folds) == 5
    assert sorted(np.concatenate(folds).tolist()) == list(range(150))
    for f in folds:
        rest = np.setdiff1d(np.arange(150), f)
        assert not set(groups[f]) & set(groups[rest])


def test_cross_validate_returns_oof_and_fold_stats():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)) * [1, 100, 0.01]
    y = 2 * X[:, 0] + 1
    res = cross_validate(lambda Xtr, ytr: LinearRegressionNP().fit(Xtr, ytr), X, y, np.arange(100), k=5)
    assert res["oof"].shape == (100,)
    assert res["r2_mean"] > 0.999 and len(res["r2_folds"]) == 5


def test_leave_element_out_holds_out_every_compound_with_element():
    sites = [("Ba", "Ti", "Sr", "Ti"), ("Sr", "Zr", "Sr", "Ti"), ("Ba", "Zr", "Ca", "Zr"), ("Ca", "Ti", "Ca", "Zr")] * 5
    X = np.arange(len(sites), dtype=float)[:, None]
    y = X[:, 0].copy()
    seen = {}

    def factory(Xtr, ytr):
        seen.setdefault("train_sizes", []).append(len(ytr))
        return LinearRegressionNP().fit(Xtr, ytr)

    res = leave_element_out(factory, X, y, sites, elements=["Ba"])
    assert res["n_test"]["Ba"] == 10 and seen["train_sizes"] == [10]


def test_permutation_importance_finds_signal_column_and_groups():
    from evaluation import permutation_importance

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = 3 * X[:, 1]
    predict = lambda Z: 3 * Z[:, 1]
    imp = permutation_importance(predict, X, y)
    assert imp.argmax() == 1 and imp[0] == 0 and imp[2] == 0
    grouped = permutation_importance(predict, X, y, column_groups=[[0, 2], [1]])
    assert grouped[0] == 0 and grouped[1] > 1
