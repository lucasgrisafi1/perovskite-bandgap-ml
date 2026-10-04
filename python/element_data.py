"""Elemental and site-specific property tables for featurization.

Two tables live here:

ELEMENT_PROPS  (v1, unchanged; used only by the baseline features)
    symbol -> (atomic_number, Pauling EN, empirical atomic radius in pm
    [Slater 1964], "valence electrons" as used in v1).

ELEMENT_INFO + SITE_PROPS  (v2 site-resolved features)
    ELEMENT_INFO[symbol] -> (Pauling EN, first ionization energy IE1 [eV],
    electron affinity EA [eV, 0 if unbound], period, group).
    Source: mendeleev 1.3.0 (Pauling EN; IE1 from NIST ASD).

    SITE_PROPS[(site, symbol)] -> dict(ox, r_ion, r_src, n_d, lone_pair)
      site       "A" (12-fold cuboctahedral) or "B" (6-fold octahedral)
      ox         oxidation state assumed on that site. A-site Ga, In, Ge,
                 Sn take their lone-pair states (Ga+, In+, Ge2+, Sn2+);
                 with these choices every one of the 1306 compounds is
                 exactly charge balanced (A + A' + B + B' = +12).
      r_ion      Shannon (1976) effective ionic radius in pm for that ion.
                 A site: CN XII where tabulated, otherwise the highest
                 tabulated CN (r_src records which). B site: CN VI.
      r_src      provenance of r_ion. "estimate" marks the three ions with
                 no Shannon entry at all (Ga+, In+, Sn2+); those values
                 are rough literature-style estimates and the pipeline's
                 --sensitivity flag checks how much they matter.
      n_d        d electrons on the ion (0 = d0, e.g. Ti4+; 10 = filled
                 d10 shell, e.g. Sn4+, Ag+). Distinguishes transition-metal
                 d0 conduction bands from s-like d10 ones.
      lone_pair  1 for ns2 lone-pair cations (Pb2+, Tl+, Sn2+, ...).

Shannon, R. D. Acta Cryst. A32, 751-767 (1976).
"""

# ── v1 table (baseline features) ────────────────────────────────────────────
ELEMENT_PROPS = {
    #        Z    EN    r(pm) val
    "Ag": (47, 1.93, 160, 11),
    "Al": (13, 1.61, 125, 3),
    "Ba": (56, 0.89, 215, 2),
    "Ca": (20, 1.00, 180, 2),
    "Cs": (55, 0.79, 260, 1),
    "Ga": (31, 1.81, 130, 3),
    "Ge": (32, 2.01, 125, 4),
    "Hf": (72, 1.30, 155, 4),
    "In": (49, 1.78, 155, 3),
    "K":  (19, 0.82, 220, 1),
    "La": (57, 1.10, 195, 3),
    "Li": (3,  0.98, 145, 1),
    "Mg": (12, 1.31, 150, 2),
    "Na": (11, 0.93, 180, 1),
    "Nb": (41, 1.60, 145, 5),
    "Pb": (82, 1.87, 180, 4),
    "Rb": (37, 0.82, 235, 1),
    "Sb": (51, 2.05, 145, 5),
    "Sc": (21, 1.36, 160, 3),
    "Si": (14, 1.90, 110, 4),
    "Sn": (50, 1.96, 145, 4),
    "Sr": (38, 0.95, 200, 2),
    "Ta": (73, 1.50, 145, 5),
    "Ti": (22, 1.54, 140, 4),
    "Tl": (81, 1.62, 190, 3),
    "V":  (23, 1.63, 135, 5),
    "Y":  (39, 1.22, 180, 3),
    "Zr": (40, 1.33, 155, 4),
    "O":  (8,  3.44, 60,  6),
}

# ── v2 element-level properties ─────────────────────────────────────────────
ELEMENT_INFO_KEYS = ("en", "ie1", "ea", "period", "group")
ELEMENT_INFO = {
    #       EN    IE1(eV) EA(eV) per grp
    "Ag": (1.93, 7.576, 1.302, 5, 11),
    "Al": (1.61, 5.986, 0.433, 3, 13),
    "Ba": (0.89, 5.212, 0.145, 6, 2),
    "Ca": (1.00, 6.113, 0.025, 4, 2),
    "Cs": (0.79, 3.894, 0.472, 6, 1),
    "Ga": (1.81, 5.999, 0.430, 4, 13),
    "Ge": (2.01, 7.899, 1.233, 4, 14),
    "Hf": (1.30, 6.825, 0.014, 6, 4),
    "In": (1.78, 5.786, 0.300, 5, 13),
    "K":  (0.82, 4.341, 0.501, 4, 1),
    "La": (1.10, 5.577, 0.470, 6, 3),
    "Li": (0.98, 5.392, 0.618, 2, 1),
    "Mg": (1.31, 7.646, 0.000, 3, 2),
    "Na": (0.93, 5.139, 0.548, 3, 1),
    "Nb": (1.60, 6.759, 0.917, 5, 5),
    "O":  (3.44, 13.618, 1.461, 2, 16),
    "Pb": (1.80, 7.417, 0.357, 6, 14),
    "Rb": (0.82, 4.177, 0.486, 5, 1),
    "Sb": (2.05, 8.608, 1.046, 5, 15),
    "Sc": (1.36, 6.561, 0.188, 4, 3),
    "Si": (1.90, 8.152, 1.390, 3, 14),
    "Sn": (1.96, 7.344, 1.112, 5, 14),
    "Sr": (0.95, 5.695, 0.048, 5, 2),
    "Ta": (1.50, 7.550, 0.322, 6, 5),
    "Ti": (1.54, 6.828, 0.079, 4, 4),
    "Tl": (1.80, 6.108, 0.377, 6, 13),
    "V":  (1.63, 6.746, 0.525, 4, 5),
    "Y":  (1.22, 6.217, 0.307, 5, 3),
    "Zr": (1.33, 6.634, 0.426, 5, 4),
}

R_O = 140  # O2- Shannon radius, CN VI (pm)


def _s(ox, r_ion, r_src, n_d, lone_pair=0):
    return {"ox": ox, "r_ion": r_ion, "r_src": r_src, "n_d": n_d, "lone_pair": lone_pair}


# ── v2 site-specific properties ─────────────────────────────────────────────
SITE_PROPS = {
    # A site (CN XII where tabulated)
    ("A", "Ag"): _s(1, 128, "Shannon CN VIII", 10),
    ("A", "Ba"): _s(2, 161, "Shannon CN XII", 0),
    ("A", "Ca"): _s(2, 134, "Shannon CN XII", 0),
    ("A", "Cs"): _s(1, 188, "Shannon CN XII", 0),
    ("A", "Ga"): _s(1, 120, "estimate", 10, 1),
    ("A", "Ge"): _s(2, 73, "Shannon CN VI", 10, 1),
    ("A", "In"): _s(1, 140, "estimate", 10, 1),
    ("A", "K"):  _s(1, 164, "Shannon CN XII", 0),
    ("A", "La"): _s(3, 136, "Shannon CN XII", 0),
    ("A", "Li"): _s(1, 92, "Shannon CN VIII", 0),
    ("A", "Mg"): _s(2, 89, "Shannon CN VIII", 0),
    ("A", "Na"): _s(1, 139, "Shannon CN XII", 0),
    ("A", "Pb"): _s(2, 149, "Shannon CN XII", 10, 1),
    ("A", "Rb"): _s(1, 172, "Shannon CN XII", 0),
    ("A", "Sn"): _s(2, 118, "estimate", 10, 1),
    ("A", "Sr"): _s(2, 144, "Shannon CN XII", 0),
    ("A", "Tl"): _s(1, 170, "Shannon CN XII", 10, 1),
    ("A", "Y"):  _s(3, 107.5, "Shannon CN IX", 0),
    # B site (CN VI)
    ("B", "Al"): _s(3, 53.5, "Shannon CN VI", 0),
    ("B", "Ga"): _s(3, 62, "Shannon CN VI", 10),
    ("B", "Ge"): _s(4, 53, "Shannon CN VI", 10),
    ("B", "Hf"): _s(4, 71, "Shannon CN VI", 0),
    ("B", "In"): _s(3, 80, "Shannon CN VI", 10),
    ("B", "Nb"): _s(5, 64, "Shannon CN VI", 0),
    ("B", "Sb"): _s(5, 60, "Shannon CN VI", 10),
    ("B", "Sc"): _s(3, 74.5, "Shannon CN VI", 0),
    ("B", "Si"): _s(4, 40, "Shannon CN VI", 0),
    ("B", "Sn"): _s(4, 69, "Shannon CN VI", 10),
    ("B", "Ta"): _s(5, 64, "Shannon CN VI", 0),
    ("B", "Ti"): _s(4, 60.5, "Shannon CN VI", 0),
    ("B", "V"):  _s(5, 54, "Shannon CN VI", 0),
    ("B", "Zr"): _s(4, 72, "Shannon CN VI", 0),
}

ESTIMATED_RADII = {key for key, v in SITE_PROPS.items() if v["r_src"] == "estimate"}


def site_prop(site, element, key):
    """Property of `element` on `site`; element-level keys fall through."""
    if key in ELEMENT_INFO_KEYS:
        return ELEMENT_INFO[element][ELEMENT_INFO_KEYS.index(key)]
    return SITE_PROPS[(site, element)][key]
