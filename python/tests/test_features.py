import numpy as np
import pytest

from features import (
    BASELINE_FEATURE_NAMES,
    SITE_FEATURE_NAMES,
    baseline_features,
    parse_formula,
    site_features,
)


def test_parse_formula():
    assert parse_formula("AgNbLaAlO6") == (["Ag", "Nb", "La", "Al", "O"], [1, 1, 1, 1, 6])


def test_baseline_rejects_unparseable_formula():
    with pytest.raises(ValueError):
        baseline_features("Ag(Nb)LaAlO6")


def test_baseline_rejects_unknown_element():
    with pytest.raises(ValueError):
        baseline_features("XxNbLaAlO6")


def test_baseline_matches_v1_definition():
    x = baseline_features("AgNbLaAlO6")
    en = np.array([1.93, 1.60, 1.10, 1.61, 3.44])
    w = np.array([1, 1, 1, 1, 6]) / 10
    assert len(BASELINE_FEATURE_NAMES) == 5
    assert x[0] == pytest.approx((w * en).sum())
    assert x[1] == pytest.approx(np.std(en))
    assert x[4] == pytest.approx(195 / 60)


def test_site_features_shape_and_names():
    x = site_features("Ag", "Nb", "La", "Al")
    assert x.shape == (len(SITE_FEATURE_NAMES),) == (49,)
    assert len(set(SITE_FEATURE_NAMES)) == 49
    assert np.all(np.isfinite(x))


def test_site_features_invariant_to_within_site_swap():
    a = site_features("Ag", "Ta", "Cs", "Nb")
    b = site_features("Cs", "Ta", "Ag", "Nb")  # a1 <-> a2
    c = site_features("Ag", "Nb", "Cs", "Ta")  # b1 <-> b2
    assert np.allclose(a, b) and np.allclose(a, c)


def test_site_features_distinguish_a_from_b():
    # Sn moved between the A and B sites -> different vector, even though
    # a site-blind composition average would treat the swap as similar.
    x1 = site_features("Ba", "Sn", "Sr", "Ti")
    x2 = site_features("Sn", "Ti", "Ba", "Sn")
    assert not np.allclose(x1, x2)


def test_tolerance_and_octahedral_factor_hand_values():
    x = site_features("Ba", "Ti", "Ba", "Ti")
    t = x[SITE_FEATURE_NAMES.index("tolerance_factor")]
    mu = x[SITE_FEATURE_NAMES.index("octahedral_factor")]
    assert t == pytest.approx((161 + 140) / (np.sqrt(2) * (60.5 + 140)))
    assert mu == pytest.approx(60.5 / 140)


def test_site_stats_values():
    x = site_features("Ba", "Ti", "Sr", "Zr")
    get = lambda n: x[SITE_FEATURE_NAMES.index(n)]
    assert get("A_en_min") == 0.89 and get("A_en_max") == 0.95
    assert get("B_r_ion_mean") == pytest.approx((60.5 + 72) / 2)
    assert get("B_r_mismatch") == pytest.approx(72 - 60.5)
    assert get("A_ox_sum") == 4
