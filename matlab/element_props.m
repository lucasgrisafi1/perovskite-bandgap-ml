function props = element_props()
%ELEMENT_PROPS Lookup struct of elemental properties for featurization.
%   props.(symbol) = [atomic_number, electronegativity_Pauling, ...
%                     atomic_radius_pm_Slater, valence_electrons]
%   Covers all 28 cations in the Pilania double-perovskite dataset + O.

props = struct();
props.Ag = [47, 1.93, 160, 11];
props.Al = [13, 1.61, 125, 3];
props.Ba = [56, 0.89, 215, 2];
props.Ca = [20, 1.00, 180, 2];
props.Cs = [55, 0.79, 260, 1];
props.Ga = [31, 1.81, 130, 3];
props.Ge = [32, 2.01, 125, 4];
props.Hf = [72, 1.30, 155, 4];
props.In = [49, 1.78, 155, 3];
props.K  = [19, 0.82, 220, 1];
props.La = [57, 1.10, 195, 3];
props.Li = [3,  0.98, 145, 1];
props.Mg = [12, 1.31, 150, 2];
props.Na = [11, 0.93, 180, 1];
props.Nb = [41, 1.60, 145, 5];
props.Pb = [82, 1.87, 180, 4];
props.Rb = [37, 0.82, 235, 1];
props.Sb = [51, 2.05, 145, 5];
props.Sc = [21, 1.36, 160, 3];
props.Si = [14, 1.90, 110, 4];
props.Sn = [50, 1.96, 145, 4];
props.Sr = [38, 0.95, 200, 2];
props.Ta = [73, 1.50, 145, 5];
props.Ti = [22, 1.54, 140, 4];
props.Tl = [81, 1.62, 190, 3];
props.V  = [23, 1.63, 135, 5];
props.Y  = [39, 1.22, 180, 3];
props.Zr = [40, 1.33, 155, 4];
props.O  = [8,  3.44, 60,  6];
end
