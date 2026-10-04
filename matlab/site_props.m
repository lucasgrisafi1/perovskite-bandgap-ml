function [info, A, B, RO] = site_props()
%SITE_PROPS Element and site-specific tables for the site-resolved features.
%   Mirrors python/element_data.py exactly (same values, same sources).
%
%   info.(el) = [EN_Pauling, IE1_eV, EA_eV, period, group]   (mendeleev 1.3.0)
%   A.(el)    = [ox, r_ion_pm, n_d, lone_pair]   A site, 12-fold
%   B.(el)    = [ox, r_ion_pm, n_d, lone_pair]   B site, 6-fold
%   RO        = 140 pm (O2-, Shannon CN VI)
%
%   r_ion: Shannon (1976) effective ionic radii. A site uses CN XII where
%   tabulated (Ag, Li, Mg: CN VIII; Y: CN IX; Ge2+: CN VI). Ga+, In+ and
%   Sn2+ have no Shannon entry: 120 / 140 / 118 pm are rough estimates
%   (see README sensitivity check). A-site Ga, In, Ge, Sn take lone-pair
%   oxidation states (1+, 1+, 2+, 2+); with these every compound in the
%   dataset is charge balanced.

info = struct();
info.Ag = [1.93, 7.576, 1.302, 5, 11];
info.Al = [1.61, 5.986, 0.433, 3, 13];
info.Ba = [0.89, 5.212, 0.145, 6, 2];
info.Ca = [1.00, 6.113, 0.025, 4, 2];
info.Cs = [0.79, 3.894, 0.472, 6, 1];
info.Ga = [1.81, 5.999, 0.430, 4, 13];
info.Ge = [2.01, 7.899, 1.233, 4, 14];
info.Hf = [1.30, 6.825, 0.014, 6, 4];
info.In = [1.78, 5.786, 0.300, 5, 13];
info.K  = [0.82, 4.341, 0.501, 4, 1];
info.La = [1.10, 5.577, 0.470, 6, 3];
info.Li = [0.98, 5.392, 0.618, 2, 1];
info.Mg = [1.31, 7.646, 0.000, 3, 2];
info.Na = [0.93, 5.139, 0.548, 3, 1];
info.Nb = [1.60, 6.759, 0.917, 5, 5];
info.O  = [3.44, 13.618, 1.461, 2, 16];
info.Pb = [1.80, 7.417, 0.357, 6, 14];
info.Rb = [0.82, 4.177, 0.486, 5, 1];
info.Sb = [2.05, 8.608, 1.046, 5, 15];
info.Sc = [1.36, 6.561, 0.188, 4, 3];
info.Si = [1.90, 8.152, 1.390, 3, 14];
info.Sn = [1.96, 7.344, 1.112, 5, 14];
info.Sr = [0.95, 5.695, 0.048, 5, 2];
info.Ta = [1.50, 7.550, 0.322, 6, 5];
info.Ti = [1.54, 6.828, 0.079, 4, 4];
info.Tl = [1.80, 6.108, 0.377, 6, 13];
info.V  = [1.63, 6.746, 0.525, 4, 5];
info.Y  = [1.22, 6.217, 0.307, 5, 3];
info.Zr = [1.33, 6.634, 0.426, 5, 4];

A = struct();          %  ox  r_ion  n_d  lone_pair
A.Ag = [1, 128,   10, 0];
A.Ba = [2, 161,    0, 0];
A.Ca = [2, 134,    0, 0];
A.Cs = [1, 188,    0, 0];
A.Ga = [1, 120,   10, 1];   % estimate
A.Ge = [2, 73,    10, 1];
A.In = [1, 140,   10, 1];   % estimate
A.K  = [1, 164,    0, 0];
A.La = [3, 136,    0, 0];
A.Li = [1, 92,     0, 0];
A.Mg = [2, 89,     0, 0];
A.Na = [1, 139,    0, 0];
A.Pb = [2, 149,   10, 1];
A.Rb = [1, 172,    0, 0];
A.Sn = [2, 118,   10, 1];   % estimate
A.Sr = [2, 144,    0, 0];
A.Tl = [1, 170,   10, 1];
A.Y  = [3, 107.5,  0, 0];

B = struct();          %  ox  r_ion  n_d  lone_pair
B.Al = [3, 53.5,  0, 0];
B.Ga = [3, 62,   10, 0];
B.Ge = [4, 53,   10, 0];
B.Hf = [4, 71,    0, 0];
B.In = [3, 80,   10, 0];
B.Nb = [5, 64,    0, 0];
B.Sb = [5, 60,   10, 0];
B.Sc = [3, 74.5,  0, 0];
B.Si = [4, 40,    0, 0];
B.Sn = [4, 69,   10, 0];
B.Ta = [5, 64,    0, 0];
B.Ti = [4, 60.5,  0, 0];
B.V  = [5, 54,    0, 0];
B.Zr = [4, 72,    0, 0];

RO = 140;
end
