import csv
import os

from element_data import ELEMENT_PROPS, SITE_PROPS, R_O, site_prop

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
        q = sum(site_prop("A", e, "ox") for e in (r["a1"], r["a2"])) + sum(
            site_prop("B", e, "ox") for e in (r["b1"], r["b2"])
        )
        assert q == 12, r["formula"]


def test_known_shannon_values():
    assert site_prop("A", "Ba", "r_ion") == 161  # Ba2+ CN XII
    assert site_prop("B", "Ti", "r_ion") == 60.5  # Ti4+ CN VI
    assert R_O == 140


def test_element_level_props_reachable_from_site():
    assert site_prop("B", "Nb", "en") == 1.60
    assert site_prop("A", "Cs", "period") == 6


def test_v1_table_unchanged():
    assert ELEMENT_PROPS["Ag"] == (47, 1.93, 160, 11)
