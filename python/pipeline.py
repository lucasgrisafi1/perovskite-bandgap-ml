"""Perovskite band gap predictor — full pipeline (numpy + matplotlib only).

Predicts GLLB-SC band gaps of 1306 double perovskite oxides (Pilania et
al., Sci. Rep. 6, 19375, 2016; matminer 'double_perovskites_gap') and
compares two feature sets:

  baseline  5 site-blind composition averages (v1 of this project)
  site      49 A-site / B-site resolved descriptors

with three from-scratch models (OLS, random forest, RBF kernel ridge)
under grouped 5-fold cross-validation and leave-one-element-out tests.

Run (from python/):
  python3 pipeline.py                 # full run, writes ../output/
  python3 pipeline.py --sensitivity   # also test the estimated-radius choice
"""

import argparse
import csv
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import element_data  # noqa: E402
from evaluation import cross_validate, group_ids, group_kfold, leave_element_out, permutation_importance  # noqa: E402
from features import BASELINE_FEATURE_NAMES, SITE_FEATURE_NAMES, SITE_PROPERTIES, baseline_features, site_features  # noqa: E402
from models import KernelRidgeRBF, LinearRegressionNP, RandomForestNP, tune_krr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "double_perovskites_gap.csv")
OUT = os.path.join(HERE, "..", "output")
SEED = 42
K_FOLDS = 5

# B-site chemistry classes used to colour the parity plot.
B_CLASS = {
    **{e: "d0 transition metal" for e in ("Sc", "Ti", "V", "Zr", "Nb", "Hf", "Ta")},
    **{e: "d10 post-transition" for e in ("Ga", "In", "Ge", "Sn", "Sb")},
    **{e: "s/p main group" for e in ("Al", "Si")},
}


# ── Models (factories see only the training fold) ───────────────────────────
def make_linear(X, y):
    return LinearRegressionNP().fit(X, y)


def make_rf(X, y):
    return RandomForestNP(n_trees=200, min_leaf=5, max_features=1 / 3, seed=SEED).fit(X, y)


def make_krr(X, y):
    alpha, gamma, _ = tune_krr(X, y, n_folds=4, seed=SEED)
    return KernelRidgeRBF(alpha=alpha, gamma=gamma).fit(X, y)


MODELS = {"Linear (OLS)": make_linear, "Random forest": make_rf, "Kernel ridge (RBF)": make_krr}


# ── Data ────────────────────────────────────────────────────────────────────
def load():
    with open(DATA) as f:
        rows = [r for r in csv.DictReader(f) if r.get("gap_gllbsc", "").strip()]
    rows = [r for r in rows if float(r["gap_gllbsc"]) > 0]
    sites = [(r["a1"], r["b1"], r["a2"], r["b2"]) for r in rows]
    y = np.array([float(r["gap_gllbsc"]) for r in rows])
    formulas = [r["formula"] for r in rows]
    return formulas, sites, y


def featurize(formulas, sites):
    X_base = np.array([baseline_features(f) for f in formulas])
    X_site = np.array([site_features(*s) for s in sites])
    assert np.isfinite(X_base).all() and np.isfinite(X_site).all()
    return {"baseline (5)": X_base, "site-resolved (49)": X_site}


def site_column_groups():
    """Group the 49 site columns into 21 interpretable families."""
    groups, labels = [], []
    for site in ("A", "B"):
        for p in SITE_PROPERTIES:
            groups.append([SITE_FEATURE_NAMES.index(f"{site}_{p}_{st}") for st in ("min", "max", "mean")])
            labels.append(f"{site}-site {p}")
    for name in SITE_FEATURE_NAMES[42:]:
        groups.append([SITE_FEATURE_NAMES.index(name)])
        labels.append(name)
    return groups, labels


# ── Outputs ─────────────────────────────────────────────────────────────────
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def parity_plot(y, oof_base, oof_site, sites, labels, path):
    classes = []
    for _, b1, _, b2 in sites:
        c1, c2 = B_CLASS[b1], B_CLASS[b2]
        classes.append(c1 if c1 == c2 else "mixed B pair")
    palette = {"d0 transition metal": "#2f6db3", "d10 post-transition": "#d9822b",
               "s/p main group": "#3a9a5b", "mixed B pair": "#8a6bb8"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
    lims = [0, 9]
    for ax, pred, title in zip(axes, (oof_base, oof_site), labels):
        for cls, col in palette.items():
            m = np.array([c == cls for c in classes])
            ax.scatter(y[m], pred[m], s=10, c=col, alpha=0.65, edgecolors="none", label=f"{cls} (n={m.sum()})")
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims), ax.set_ylim(lims), ax.set_aspect("equal")
        ax.set_xlabel("GLLB-SC band gap (eV)")
        ax.set_ylabel("Predicted band gap, out-of-fold (eV)")
        ax.set_title(title, fontsize=10)
    axes[1].legend(fontsize=8, loc="upper left", title="B-site cations", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def importance_plot(imp, std, labels, path):
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor="white")
    ax.barh([labels[i] for i in order], imp[order], xerr=std[order], color="#3a8f6a", ecolor="#555", capsize=2)
    ax.set_xlabel("Increase in held-out RMSE when permuted (eV), mean ± sd over 5 folds")
    ax.set_title("Permutation importance — random forest, site-resolved features", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def loeo_plot(loeo_base, loeo_site, path):
    els = sorted(loeo_site["rmse_by_element"], key=lambda e: loeo_site["rmse_by_element"][e])
    x = np.arange(len(els))
    fig, ax = plt.subplots(figsize=(11, 4), facecolor="white")
    ax.bar(x - 0.2, [loeo_base["rmse_by_element"][e] for e in els], 0.4, label="baseline (5)", color="#b0b0b0")
    ax.bar(x + 0.2, [loeo_site["rmse_by_element"][e] for e in els], 0.4, label="site-resolved (49)", color="#2f6db3")
    ax.set_xticks(x, [f"{e}\n{loeo_site['n_test'][e]}" for e in els], fontsize=8)
    ax.set_ylabel("RMSE on held-out compounds (eV)")
    ax.set_title("Leave-one-element-out (random forest): error on compounds containing an element never seen in training"
                 " (number = compounds held out)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def run_sensitivity(formulas, sites, y, groups):
    print("\nSensitivity: estimated A-site radii (Ga+, In+, Sn2+) scaled by 0.85 / 1.00 / 1.15")
    original = {k: element_data.SITE_PROPS[k]["r_ion"] for k in element_data.ESTIMATED_RADII}
    rows = []
    try:
        for scale in (0.85, 1.0, 1.15):
            for k, r in original.items():
                element_data.SITE_PROPS[k]["r_ion"] = r * scale
            X = np.array([site_features(*s) for s in sites])
            res = cross_validate(make_rf, X, y, groups, K_FOLDS, SEED)
            rows.append([scale, f"{res['r2_mean']:.4f}", f"{res['rmse_mean']:.4f}"])
            print(f"  scale {scale:.2f}: R2 {res['r2_mean']:.3f}  RMSE {res['rmse_mean']:.3f} eV")
    finally:
        for k, r in original.items():
            element_data.SITE_PROPS[k]["r_ion"] = r
    write_csv(os.path.join(OUT, "sensitivity_estimated_radii.csv"), ["radius_scale", "RF_R2_mean", "RF_RMSE_mean_eV"], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensitivity", action="store_true", help="also run the estimated-radius sensitivity check")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    formulas, sites, y = load()
    groups = group_ids(sites)
    feature_sets = featurize(formulas, sites)
    print(f"Compounds: {len(y)} (gap > 0)  |  distinct materials (CV groups): {len(np.unique(groups))}")
    print(f"Band gap: {y.min():.2f}-{y.max():.2f} eV, mean {y.mean():.2f}, sd {y.std(ddof=1):.2f}")

    # Shared with MATLAB: exact features (for a cross-check) and fold ids
    # (so both implementations are scored on identical splits).
    write_csv(os.path.join(OUT, "site_features_python.csv"), ["formula", *SITE_FEATURE_NAMES],
              [[f, *[repr(float(v)) for v in row]] for f, row in zip(formulas, feature_sets["site-resolved (49)"])])
    fold_of = np.empty(len(y), dtype=int)
    for k, te in enumerate(group_kfold(groups, K_FOLDS, SEED)):
        fold_of[te] = k + 1
    write_csv(os.path.join(OUT, "cv_folds.csv"), ["formula", "group", "fold"],
              [[f, g + 1, k] for f, g, k in zip(formulas, groups, fold_of)])

    # 1. Grouped 5-fold CV
    print(f"\nGrouped {K_FOLDS}-fold cross-validation (mean ± sd over folds)")
    print(f"{'Features':<20}{'Model':<20}{'R2':>14}{'RMSE (eV)':>16}{'MAE (eV)':>16}")
    cv, cv_rows = {}, []
    fam_groups, fam_labels = site_column_groups()
    fold_importances = []

    def collect_importance(model, X_te, y_te):
        fold_importances.append(permutation_importance(model.predict, X_te, y_te, fam_groups, n_repeats=5, seed=SEED))

    for fs, X in feature_sets.items():
        for mname, factory in MODELS.items():
            hook = collect_importance if (fs.startswith("site") and mname == "Random forest") else None
            r = cross_validate(factory, X, y, groups, K_FOLDS, SEED, on_fold=hook)
            cv[(fs, mname)] = r
            print(f"{fs:<20}{mname:<20}{r['r2_mean']:>8.3f} ± {r['r2_std']:.3f}"
                  f"{r['rmse_mean']:>9.3f} ± {r['rmse_std']:.3f}{r['mae_mean']:>9.3f} ± {r['mae_std']:.3f}")
            cv_rows.append([fs, mname] + [f"{r[k]:.4f}" for k in
                            ("r2_mean", "r2_std", "rmse_mean", "rmse_std", "mae_mean", "mae_std")])
    write_csv(os.path.join(OUT, "cv_results.csv"),
              ["features", "model", "R2_mean", "R2_sd", "RMSE_mean_eV", "RMSE_sd_eV", "MAE_mean_eV", "MAE_sd_eV"], cv_rows)

    # 2. Leave-one-element-out
    print("\nLeave-one-element-out (pooled over all 28 held-out cations)")
    loeo, loeo_rows = {}, []
    for fs, X in feature_sets.items():
        for mname in ("Random forest", "Kernel ridge (RBF)"):
            r = leave_element_out(MODELS[mname], X, y, sites)
            loeo[(fs, mname)] = r
            print(f"{fs:<20}{mname:<20}R2 {r['r2']:.3f}  RMSE {r['rmse']:.3f} eV  MAE {r['mae']:.3f} eV")
            loeo_rows.append([fs, mname, f"{r['r2']:.4f}", f"{r['rmse']:.4f}", f"{r['mae']:.4f}"])
    write_csv(os.path.join(OUT, "loeo_results.csv"), ["features", "model", "R2", "RMSE_eV", "MAE_eV"], loeo_rows)
    lb, ls = loeo[("baseline (5)", "Random forest")], loeo[("site-resolved (49)", "Random forest")]
    write_csv(os.path.join(OUT, "loeo_rmse_by_element.csv"),
              ["element", "n_test", "RMSE_baseline_RF_eV", "RMSE_site_RF_eV"],
              [[e, ls["n_test"][e], f"{lb['rmse_by_element'][e]:.4f}", f"{ls['rmse_by_element'][e]:.4f}"]
               for e in sorted(ls["n_test"])])

    # 3. Importance + figures
    imp = np.array(fold_importances)
    imp_mean, imp_sd = imp.mean(0), imp.std(0, ddof=1)
    order = np.argsort(-imp_mean)
    write_csv(os.path.join(OUT, "feature_importance_site.csv"), ["feature_family", "dRMSE_mean_eV", "dRMSE_sd_eV"],
              [[fam_labels[i], f"{imp_mean[i]:.4f}", f"{imp_sd[i]:.4f}"] for i in order])
    print("\nTop permutation-importance families (RF, site features):")
    for i in order[:6]:
        print(f"  {fam_labels[i]:<22} +{imp_mean[i]:.3f} ± {imp_sd[i]:.3f} eV")

    best_site = max((k for k in cv if k[0].startswith("site")), key=lambda k: cv[k]["r2_mean"])
    rb, rs = cv[("baseline (5)", "Random forest")], cv[best_site]
    parity_plot(y, rb["oof"], rs["oof"], sites,
                [f"Baseline features, random forest\nR² = {rb['r2_mean']:.3f}, RMSE = {rb['rmse_mean']:.2f} eV",
                 f"Site-resolved features, {best_site[1].lower()}\nR² = {rs['r2_mean']:.3f}, RMSE = {rs['rmse_mean']:.2f} eV"],
                os.path.join(OUT, "parity_baseline_vs_site.png"))
    importance_plot(imp_mean, imp_sd, fam_labels, os.path.join(OUT, "feature_importance_site.png"))
    loeo_plot(lb, ls, os.path.join(OUT, "loeo_rmse_by_element.png"))

    if args.sensitivity:
        run_sensitivity(formulas, sites, y, groups)

    print(f"\nWrote results to output/  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
