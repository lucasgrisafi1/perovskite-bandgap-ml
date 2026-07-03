"""Element property lookup for compositional feature engineering.

Columns: atomic_number, electronegativity (Pauling), atomic_radius_pm
(empirical, Slater 1964), valence_electrons (outer s+p, s+d for TMs).
Covers all 28 cations in the Pilania double-perovskite dataset + O.
"""

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
