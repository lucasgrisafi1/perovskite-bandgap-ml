function [x, names] = site_features(a1, b1, a2, b2, info, A, B, RO)
%SITE_FEATURES 49 A/B-site-resolved descriptors (mirrors python/features.py).
%   x = SITE_FEATURES(a1, b1, a2, b2, info, A, B, RO) for compound
%   a1 b1 a2 b2 O6, with tables from SITE_PROPS. For each site (A, B) and
%   property (EN, r_ion, IE1, EA, period, group, n_d): min, max, mean over
%   the two cations, then 7 structural/charge descriptors.

props = {'en', 'r_ion', 'ie1', 'ea', 'period', 'group', 'n_d'};
stats = {'min', 'max', 'mean'};
EN_O = 3.44;

x = zeros(1, 49);
names = cell(1, 49);
k = 0;
sites = {'A', 'B'};
pairs = {{a1, a2}, {b1, b2}};
for s = 1:2
    for p = 1:numel(props)
        v = [prop(sites{s}, pairs{s}{1}, props{p}), prop(sites{s}, pairs{s}{2}, props{p})];
        sv = [min(v), max(v), mean(v)];
        for st = 1:3
            k = k + 1;
            x(k) = sv(st);
            names{k} = sprintf('%s_%s_%s', sites{s}, props{p}, stats{st});
        end
    end
end

rA = mean([prop('A', a1, 'r_ion'), prop('A', a2, 'r_ion')]);
rB_each = [prop('B', b1, 'r_ion'), prop('B', b2, 'r_ion')];
rB = mean(rB_each);
x(43) = (rA + RO) / (sqrt(2) * (rB + RO));                         % tolerance factor
x(44) = rB / RO;                                                   % octahedral factor
x(45) = prop('A', a1, 'ox') + prop('A', a2, 'ox');                 % A ox. state sum
x(46) = prop('A', a1, 'lone_pair') + prop('A', a2, 'lone_pair');   % lone pairs on A
x(47) = abs(rB_each(1) - rB_each(2));                              % B size mismatch
x(48) = EN_O - mean([prop('B', b1, 'en'), prop('B', b2, 'en')]);   % EN(O) - EN(B)
x(49) = EN_O - mean([prop('A', a1, 'en'), prop('A', a2, 'en')]);   % EN(O) - EN(A)
names(43:49) = {'tolerance_factor', 'octahedral_factor', 'A_ox_sum', ...
    'A_lone_pair_count', 'B_r_mismatch', 'dEN_O_minus_B', 'dEN_O_minus_A'};

    function v = prop(site, el, key)
        element_keys = {'en', 'ie1', 'ea', 'period', 'group'};
        site_keys = {'ox', 'r_ion', 'n_d', 'lone_pair'};
        j = find(strcmp(element_keys, key), 1);
        if ~isempty(j)
            v = info.(el)(j);
            return;
        end
        if strcmp(site, 'A'), tbl = A; else, tbl = B; end
        if ~isfield(tbl, el)
            error('site_features:noData', 'No %s-site data for %s.', site, el);
        end
        v = tbl.(el)(strcmp(site_keys, key));
    end
end
