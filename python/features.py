"""Feature construction for A1B1A2B2O6 double perovskites.

baseline_features(formula)
    The original (v1) 5 composition-averaged descriptors, kept unchanged
    for comparison. They are site-blind: every atom, including the six
    oxygens, is pooled into one average.

site_features(a1, b1, a2, b2)
    49 descriptors that keep the A site (12-fold) and B site (6-fold)
    separate. For each site and each property, the min / max / mean over
    the two cations on that site (order-invariant within a site), plus
    structural and charge descriptors.
"""

import re

import numpy as np

from element_data import ELEMENT_PROPS, R_O, site_prop

# ── Formula parsing ─────────────────────────────────────────────────────────


def parse_formula(formula):
    """'AgNbLaAlO6' -> (['Ag', 'Nb', 'La', 'Al', 'O'], [1, 1, 1, 1, 6])"""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    elements = [sym for sym, _ in tokens]
    counts = [int(num) if num else 1 for _, num in tokens]
    return elements, counts


# ── Baseline (v1) features ──────────────────────────────────────────────────

BASELINE_FEATURE_NAMES = ["Mean EN", "Std EN", "Mean Radius", "Mean Valence", "Radius Ratio"]


def baseline_features(formula):
    """v1 descriptors: stoichiometry-weighted means over all atoms."""
    elements, counts = parse_formula(formula)
    consumed = "".join(e + (str(c) if c > 1 else "") for e, c in zip(elements, counts))
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
        np.sum(w * en),               # mean electronegativity
        np.std(en),                   # EN spread over distinct elements (unweighted)
        np.sum(w * radius),           # mean atomic radius
        np.sum(w * valence),          # mean valence electrons
        radius.max() / radius.min(),  # largest radius / O radius (O is always smallest)
    ])


# ── Site-resolved features ──────────────────────────────────────────────────

SITE_PROPERTIES = ("en", "r_ion", "ie1", "ea", "period", "group", "n_d")
_STATS = ("min", "max", "mean")
EN_O = 3.44

SITE_FEATURE_NAMES = (
    [f"{s}_{p}_{st}" for s in ("A", "B") for p in SITE_PROPERTIES for st in _STATS]
    + [
        "tolerance_factor",    # Goldschmidt t = (rA + rO) / (sqrt2 (rB + rO))
        "octahedral_factor",   # mu = rB / rO
        "A_ox_sum",            # A + A' oxidation states (B + B' = 12 - this)
        "A_lone_pair_count",   # number of ns2 lone-pair cations on A
        "B_r_mismatch",        # |r_B1 - r_B2|, B-site size disorder
        "dEN_O_minus_B",       # EN(O) - mean EN(B): B-O bond ionicity
        "dEN_O_minus_A",       # EN(O) - mean EN(A): A-O bond ionicity
    ]
)


def _site_values(site, pair, prop):
    return np.array([site_prop(site, e, prop) for e in pair], dtype=float)


def site_features(a1, b1, a2, b2):
    """49 site-resolved descriptors for compound a1 b1 a2 b2 O6."""
    A, B = (a1, a2), (b1, b2)
    x = []
    for site, pair in (("A", A), ("B", B)):
        for prop in SITE_PROPERTIES:
            v = _site_values(site, pair, prop)
            x += [v.min(), v.max(), v.mean()]

    rA = _site_values("A", A, "r_ion").mean()
    rB_each = _site_values("B", B, "r_ion")
    rB = rB_each.mean()
    x += [
        (rA + R_O) / (np.sqrt(2) * (rB + R_O)),
        rB / R_O,
        _site_values("A", A, "ox").sum(),
        _site_values("A", A, "lone_pair").sum(),
        abs(rB_each[0] - rB_each[1]),
        EN_O - _site_values("B", B, "en").mean(),
        EN_O - _site_values("A", A, "en").mean(),
    ]
    return np.array(x, dtype=float)
