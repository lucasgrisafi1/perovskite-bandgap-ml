function fold = group_folds(groupKey, k, seed)
%GROUP_FOLDS Assign rows to k CV folds so that no group spans two folds.
%   fold = GROUP_FOLDS(groupKey, k, seed). groupKey is a cellstr (one key
%   per row). Groups are shuffled with rng(seed) and assigned greedily to
%   the currently smallest fold. Used only when ../output/cv_folds.csv
%   (written by the Python pipeline) is missing; the Python fold file is
%   preferred so both implementations score identical splits.

rng(seed);
[~, ~, g] = unique(groupKey);
nG = max(g);
order = randperm(nG);
fold = zeros(numel(g), 1);
sizes = zeros(1, k);
for gi = order
    [~, f] = min(sizes);
    members = (g == gi);
    fold(members) = f;
    sizes(f) = sizes(f) + sum(members);
end
end
