function x = baseline_features(formula, props)
%BASELINE_FEATURES The original (v1) 5 site-blind composition averages.
%   x = BASELINE_FEATURES('AgNbLaAlO6', element_props()) returns
%   [mean EN, std EN, mean radius, mean valence, max/min radius].
%   Kept unchanged for comparison with the site-resolved features.

[elements, counts] = parse_formula(formula);
EN = zeros(1, numel(elements));
radius = EN;
valence = EN;
for j = 1:numel(elements)
    if ~isfield(props, elements{j})
        error('baseline_features:noData', 'No element data for ''%s'' in ''%s''.', elements{j}, formula);
    end
    ep = props.(elements{j});
    EN(j) = ep(2);
    radius(j) = ep(3);
    valence(j) = ep(4);
end
w = counts / sum(counts);
x = [sum(w .* EN), std(EN, 1), sum(w .* radius), sum(w .* valence), max(radius) / min(radius)];
end
