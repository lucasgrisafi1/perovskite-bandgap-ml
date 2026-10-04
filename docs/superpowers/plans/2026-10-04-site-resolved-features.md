# Site-Resolved Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace site-blind averaged descriptors with A/B-site-resolved chemistry, evaluate with grouped CV + leave-one-element-out, mirror in MATLAB, and rewrite the README so every number is reproducible.

**Architecture:** Split `python/bandgap_pipeline.py` into `element_data` → `features` → `models` → `evaluation` → `pipeline`. Pure numpy models. MATLAB mirrors the same feature definitions and folds.

**Tech Stack:** Python 3, numpy, matplotlib, pytest (dev only); MATLAB R2021a+ with Statistics and ML Toolbox.

Spec: `docs/superpowers/specs/2026-10-04-site-resolved-features-design.md`

Run all tests from repo root: `python3 -m pytest python/tests -q`

---

### Task 1: Element data

**Files:** Modify `python/element_data.py`; Test `python/tests/test_element_data.py`

- [ ] Step 1: Write failing tests

```python
import csv, os
from element_data import ELEMENT_PROPS, SITE_PROPS, site_prop, R_O

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "double_perovskites_gap.csv")

def rows():
    with open(DATA) as f:
        return list(csv.DictReader(f))

def test_every_dataset_cation_has_site_data():
    for r in rows():
        for e in (r["a1"], r["a2"]):
            assert ("A", e) in SITE_PROPS
        for e in (r["b1"], r["b2"]):
            assert ("B", e) in SITE_PROPS

def test_site_oxidation_states_charge_balance_all_rows():
    for r in rows():
        q = sum(site_prop("A", e, "ox") for e in (r["a1"], r["a2"])) + \
            sum(site_prop("B", e, "ox") for e in (r["b1"], r["b2"]))
        assert q == 12, r["formula"]

def test_known_shannon_values():
    assert site_prop("A", "Ba", "r_ion") == 161   # Ba2+ CN XII
    assert site_prop("B", "Ti", "r_ion") == 60.5  # Ti4+ CN VI
    assert R_O == 140

def test_v1_table_unchanged():
    assert ELEMENT_PROPS["Ag"] == (47, 1.93, 160, 11)
```

- [ ] Step 2: `python3 -m pytest python/tests/test_element_data.py -q` → ImportError (SITE_PROPS missing)
- [ ] Step 3: Add `ELEMENT_INFO` (EN, IE1, EA, period, group per element, from mendeleev/NIST), `SITE_PROPS[(site, el)] = dict(ox, r_ion, r_src, n_d, lone_pair)`, `site_prop(site, el, key)` that falls back to `ELEMENT_INFO`, `R_O = 140`, `ESTIMATED_RADII` set.
- [ ] Step 4: tests pass
- [ ] Step 5: commit `feat(data): site-resolved ionic radii and oxidation states`

### Task 2: Features

**Files:** Create `python/features.py`; Test `python/tests/test_features.py`

- [ ] Step 1: failing tests

```python
import numpy as np, pytest
from features import parse_formula, baseline_features, site_features, SITE_FEATURE_NAMES, BASELINE_FEATURE_NAMES

def test_parse_formula():
    assert parse_formula("AgNbLaAlO6") == (["Ag", "Nb", "La", "Al", "O"], [1, 1, 1, 1, 6])

def test_parse_formula_rejects_garbage():
    with pytest.raises(ValueError):
        baseline_features("Ag(Nb)LaAlO6")

def test_baseline_matches_v1_values():
    x = baseline_features("AgNbLaAlO6")
    en = np.array([1.93, 1.60, 1.10, 1.61, 3.44]); w = np.array([1,1,1,1,6])/10
    assert x[0] == pytest.approx((w*en).sum())
    assert x[4] == pytest.approx(195/60)

def test_site_features_shape_and_names():
    x = site_features("Ag", "Nb", "La", "Al")
    assert x.shape == (len(SITE_FEATURE_NAMES),) == (49,)

def test_site_features_invariant_to_within_site_swap():
    a = site_features("Ag", "Ta", "Cs", "Nb")
    b = site_features("Cs", "Ta", "Ag", "Nb")   # a1<->a2
    c = site_features("Ag", "Nb", "Cs", "Ta")   # b1<->b2
    assert np.allclose(a, b) and np.allclose(a, c)

def test_tolerance_factor_hand_value():
    x = site_features("Ba", "Ti", "Ba", "Ti")
    t = x[SITE_FEATURE_NAMES.index("tolerance_factor")]
    assert t == pytest.approx((161 + 140) / (np.sqrt(2) * (60.5 + 140)))
```

- [ ] Step 2: run → ImportError
- [ ] Step 3: implement `features.py` (move parse/featurize from v1; add site_features per spec)
- [ ] Step 4: pass
- [ ] Step 5: commit `feat(features): site-resolved descriptors with tolerance/octahedral factors`

### Task 3: Models

**Files:** Create `python/models.py` (moved from v1 + `max_features`); Test `python/tests/test_models.py`

- [ ] Step 1: tests — linear recovers exact coefficients on y = 2x0 − x1 + 3; RF on y = step(x0) gets R² > 0.9 on train; KRR interpolates sin(x) with RMSE < 0.05; `tune_krr` returns a grid value.
- [ ] Step 2: fail → Step 3: implement → Step 4: pass
- [ ] Step 5: commit `refactor(models): move numpy models to models.py; RF max_features`

### Task 4: Evaluation

**Files:** Create `python/evaluation.py`; Test `python/tests/test_evaluation.py`

- [ ] Step 1: tests

```python
import numpy as np
from evaluation import compute_metrics, group_ids, group_kfold, cross_validate

def test_metrics_perfect():
    r2, rmse, mae = compute_metrics(np.array([1., 2, 3]), np.array([1., 2, 3]))
    assert (r2, rmse, mae) == (1.0, 0.0, 0.0)

def test_group_ids_merge_site_relabelled_duplicates():
    g = group_ids([("Ag","Ta","Cs","Nb"), ("Cs","Ta","Ag","Nb"), ("Ag","Nb","Cs","Ta"), ("Ba","Ti","Sr","Ti")])
    assert g[0] == g[1] == g[2] != g[3]

def test_group_kfold_partitions_without_splitting_groups():
    groups = np.repeat(np.arange(50), 3)
    folds = group_kfold(groups, k=5, seed=0)
    seen = np.concatenate(folds)
    assert sorted(seen) == list(range(150))
    for f in folds:
        assert not set(groups[f]) & set(np.setdiff1d(groups, groups[f]))

def test_cross_validate_scales_inside_fold_and_returns_oof():
    rng = np.random.default_rng(0); X = rng.normal(size=(100, 3)); y = X[:, 0] * 2
    from models import LinearRegressionNP
    res = cross_validate(lambda: LinearRegressionNP(), X, y, np.arange(100), k=5)
    assert res["oof"].shape == (100,) and res["r2_mean"] > 0.99
```

- [ ] Step 2: fail → Step 3: implement (+ `leave_element_out`, `permutation_importance`) → Step 4: pass
- [ ] Step 5: commit `feat(eval): grouped K-fold CV and leave-one-element-out`

### Task 5: Pipeline + outputs

**Files:** Create `python/pipeline.py`; Delete `python/bandgap_pipeline.py`; regenerate `output/`

- [ ] Step 1: implement pipeline per spec (KRR tuned inside each fold via `model_factory` that calls `tune_krr` on the fold's training data)
- [ ] Step 2: `cd python && python3 pipeline.py` → prints CV table, LOEO table; writes CSVs + PNGs
- [ ] Step 3: radius-estimate sensitivity: `python3 pipeline.py --sensitivity` scales Ga⁺/In⁺/Sn²⁺ radii by 0.85/1.15, prints RF CV R²
- [ ] Step 4: remove stale v1 outputs; commit `feat: pipeline with grouped CV, LOEO, new figures`

### Task 6: MATLAB mirror

**Files:** Modify `matlab/element_props.m`, `matlab/main.m`; Create `matlab/site_features.m`, `matlab/baseline_features.m`, `matlab/group_folds.m`

- [ ] Step 1: port tables/feature math 1:1 (same order as `SITE_FEATURE_NAMES`)
- [ ] Step 2: cross-check: Python writes `output/site_features_python.csv`; MATLAB `main.m` asserts its own matrix matches it to 1e-9 when the file exists
- [ ] Step 3: commit `feat(matlab): mirror site features and grouped CV`

### Task 7: README

- [ ] Rewrite from verified outputs only; cite Pilania et al. 2016 numbers (KRR, 90/10 bootstrap: 0.50 eV / R² 0.90 with 4-D descriptor; ~0.37 eV / ~0.94 with 16-D) and the protocol difference.
- [ ] Commit `docs: rewrite README with verified results`

### Task 8: Verify + ship

- [ ] Full test run, clean rerun of pipeline, diff README numbers against CSVs programmatically.
- [ ] Push branch, open PR.
