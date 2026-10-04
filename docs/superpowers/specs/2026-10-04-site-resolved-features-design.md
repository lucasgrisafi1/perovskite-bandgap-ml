# Site-resolved features & honest evaluation — design

Date: 2026-10-04 · Branch: `feature/site-resolved-features`

## Problem

The v1 pipeline averages elemental properties over every atom in
A1B1A2B2O6. This erases which cation sits on the A site (12-fold
cuboctahedral) and which on the B site (6-fold octahedral), although in
these oxides the conduction-band minimum is set mainly by B-site cation
states and the A site tunes the lattice. Consequences observed:

- R² ≈ 0.54 / RMSE ≈ 1.03 eV ceiling for all three models.
- Oxygen (fixed 6/10 weight, always the smallest atom) makes three
  features affine transforms of cation means and turns "radius ratio"
  into "largest cation radius / 60 pm".
- 90 pairs of site-relabelled duplicates (e.g. AgTaCsNbO6 /
  CsTaAgNbO6, gaps equal to 1 meV) can straddle the train/test split.
- Single 80/20 split: R² varies ±0.02–0.03 between seeds.
- README result tables predate the leakage fix and disagree with
  `output/*.csv`.

## Decisions (agreed with user)

- Python (numpy + matplotlib only, models from scratch) is the verified
  primary implementation; every README number is reproduced by it.
- MATLAB mirrors the new features and evaluation; README labels its
  numbers "reproduce with main.m" until run.
- Deliver via feature branch + PR.

## Architecture (python/)

| File | Responsibility |
|------|----------------|
| `element_data.py` | Elemental table (existing v1 columns + Shannon radii per site, site oxidation states, IE1, EA, period, group, ionic d-count, lone-pair flag), with sources. |
| `features.py` | `parse_formula`, `baseline_features(formula)` (v1, unchanged math), `site_features(a1,b1,a2,b2)` + `SITE_FEATURE_NAMES`. |
| `models.py` | `LinearRegressionNP`, `RandomForestNP` (adds `max_features`), `KernelRidgeRBF`, `tune_krr`. |
| `evaluation.py` | `compute_metrics`, `group_ids`, `group_kfold`, `cross_validate`, `leave_element_out`, `permutation_importance`. Standardization fitted inside each training fold. |
| `pipeline.py` | Orchestrates: load → featurize both sets → grouped 5-fold CV for 3 models × 2 feature sets → leave-one-element-out → figures + CSVs. |
| `bandgap_pipeline.py` | Removed (superseded by `pipeline.py`). |
| `tests/` | pytest unit tests. |

## Chemistry conventions

- Site oxidation states: A-site Ga⁺, In⁺, Ge²⁺, Sn²⁺ (lone-pair
  states); all others at their usual state (alkali 1+, alkaline earth
  2+, La/Y 3+, Ag⁺, Tl⁺, Pb²⁺; B-site Al/Ga/In/Sc 3+, Si/Ge/Sn/Ti/Zr/Hf
  4+, V/Nb/Ta/Sb 5+). With these, all 1306 compounds are exactly
  charge-balanced (A+A'+B+B' = 12) — verified in a test.
- Radii: Shannon (1976) effective ionic radii; A-site uses CN XII when
  tabulated, else the highest tabulated CN for that ion; B-site uses CN
  VI; O²⁻ = 140 pm. Ga⁺, In⁺, Sn²⁺ have no Shannon entry: flagged
  estimates, with a sensitivity check (±15 %) reported in the README.
- Tolerance factor t = (r̄A + rO) / (√2 (r̄B + rO)); octahedral factor
  μ = r̄B / rO.

## Site feature vector (order fixed in `SITE_FEATURE_NAMES`)

For site S ∈ {A, B} and property p ∈ {EN, r_ion, IE1, EA, period,
group, d-count}: min, max, mean over the two cations (42 values,
invariant to swapping cations within a site). Plus: t, μ, A-site
oxidation-state sum, A-site lone-pair count, |r_B1 − r_B2|,
EN(O) − mean EN(B), EN(O) − mean EN(A). Total 49.

## Evaluation protocol

- Group key = (sorted A pair, sorted B pair); site-relabelled
  duplicates share a group.
- 5-fold grouped CV (seeded shuffle of groups). Report mean ± std of
  per-fold R², RMSE, MAE plus pooled out-of-fold predictions for plots.
- KRR hyperparameters tuned by inner 4-fold CV on each training fold.
- Leave-one-element-out (LOEO): for each of 28 cations, hold out every
  compound containing it; pooled R²/RMSE. Run for both feature sets
  with RF.
- Permutation importance on the out-of-fold RF, site features.

## Outputs (output/)

`cv_results.csv`, `loeo_results.csv`, `parity_site_features.png`
(coloured by B-site chemistry class), `baseline_vs_site.png`,
`feature_importance_site.png`. v1 figures removed or regenerated.

## MATLAB

`element_props.m` gains the same columns; `site_features.m`,
`group_folds.m`; `main.m` runs grouped 5-fold CV for fitlm / TreeBagger
/ fitrsvm (KernelScale 'auto') on both feature sets and writes
`output/cv_results_matlab.csv`.

## Testing

parse_formula (valid/invalid), baseline features unchanged vs v1 for a
known formula, site features invariant to within-site swaps, tolerance
factor hand-check, charge balance for all rows, group_kfold never
splits a group and covers every row once, metrics on toy data, RF/KRR
fit-predict smoke tests.

## Out of scope

One-hot site encodings, GNNs, external ML libraries, new datasets.
