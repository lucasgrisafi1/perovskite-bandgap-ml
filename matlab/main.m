%% Perovskite band gap predictor — MATLAB implementation
%  Predicts GLLB-SC band gaps of 1306 double perovskite oxides (Pilania et
%  al., Sci. Rep. 6, 19375, 2016; matminer 'double_perovskites_gap') and
%  compares two feature sets with fitlm, TreeBagger and fitrsvm:
%
%    baseline  5 site-blind composition averages (v1)
%    site      49 A-site / B-site resolved descriptors
%
%  Mirrors python/pipeline.py (the verified reference). If the Python
%  pipeline has been run, this script (1) checks its site features match
%  Python's to 1e-9 and (2) reuses Python's CV folds so the numbers are
%  directly comparable.
%
%  Requires: Statistics and Machine Learning Toolbox (R2021a+).
%  Run from the matlab/ folder. Outputs go to ../output/.

clear; clc; close all;
outDir = fullfile('..', 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end
K = 5;
SEED = 42;

%% 1. Load data
T = readtable(fullfile('..', 'data', 'double_perovskites_gap.csv'), 'TextType', 'char');
T = T(~isnan(T.gap_gllbsc) & T.gap_gllbsc > 0, :);
y = T.gap_gllbsc;
n = height(T);
fprintf('Compounds: %d (gap > 0)\n', n);

%% 2. Features
props = element_props();
[info, Atab, Btab, RO] = site_props();
Xbase = zeros(n, 5);
Xsite = zeros(n, 49);
for i = 1:n
    Xbase(i, :) = baseline_features(T.formula{i}, props);
    [Xsite(i, :), siteNames] = site_features(T.a1{i}, T.b1{i}, T.a2{i}, T.b2{i}, info, Atab, Btab, RO);
end

pyFeat = fullfile(outDir, 'site_features_python.csv');
if exist(pyFeat, 'file')
    P = readtable(pyFeat, 'TextType', 'char');
    assert(isequal(P.formula, T.formula), 'Row order differs from Python feature file.');
    maxDiff = max(abs(table2array(P(:, 2:end)) - Xsite), [], 'all');
    assert(maxDiff < 1e-9, 'Site features differ from Python (max |diff| = %g).', maxDiff);
    fprintf('Site features match Python to %.1e\n', maxDiff);
end

%% 3. Grouped folds (same A pair + same B pair -> same group; see README)
pyFolds = fullfile(outDir, 'cv_folds.csv');
if exist(pyFolds, 'file')
    F = readtable(pyFolds, 'TextType', 'char');
    assert(isequal(F.formula, T.formula), 'Row order differs from Python fold file.');
    fold = F.fold;
    fprintf('Using Python CV folds (%s)\n', pyFolds);
else
    key = cell(n, 1);
    for i = 1:n
        a = sort({T.a1{i}, T.a2{i}});
        b = sort({T.b1{i}, T.b2{i}});
        key{i} = strjoin([a, {'|'}, b], ',');
    end
    fold = group_folds(key, K, SEED);
    fprintf('Python fold file not found; built grouped folds in MATLAB\n');
end

%% 4. Grouped K-fold CV: 3 models x 2 feature sets
featureSets = {Xbase, Xsite};
fsNames = {'baseline (5)', 'site-resolved (49)'};
modelNames = {'Linear (fitlm)', 'Random forest (TreeBagger)', 'SVM RBF (fitrsvm)'};
rows = {};
oof = cell(2, 3);
fprintf('\n%-20s %-28s %16s %18s\n', 'Features', 'Model', 'R2', 'RMSE (eV)');
for s = 1:2
    X = featureSets{s};
    for m = 1:3
        pred = zeros(n, 1);
        r2 = zeros(K, 1); rmse = r2; mae = r2;
        for k = 1:K
            te = (fold == k); tr = ~te;
            [Xtr, Xte] = standardize_train(X(tr, :), X(te, :));
            pred(te) = fit_predict(m, Xtr, y(tr), Xte, SEED);
            [r2(k), rmse(k), mae(k)] = metrics(y(te), pred(te));
        end
        oof{s, m} = pred;
        fprintf('%-20s %-28s %7.3f +- %.3f %8.3f +- %.3f\n', fsNames{s}, modelNames{m}, ...
            mean(r2), std(r2), mean(rmse), std(rmse));
        rows(end + 1, :) = {fsNames{s}, modelNames{m}, mean(r2), std(r2), mean(rmse), std(rmse), mean(mae), std(mae)}; %#ok<SAGROW>
    end
end
cvTable = cell2table(rows, 'VariableNames', ...
    {'features', 'model', 'R2_mean', 'R2_sd', 'RMSE_mean_eV', 'RMSE_sd_eV', 'MAE_mean_eV', 'MAE_sd_eV'});
writetable(cvTable, fullfile(outDir, 'cv_results_matlab.csv'));

%% 5. Leave-one-element-out (TreeBagger)
elements = unique([T.a1; T.a2; T.b1; T.b2]);
loeoRows = {};
fprintf('\nLeave-one-element-out, random forest (pooled over %d cations)\n', numel(elements));
for s = 1:2
    X = featureSets{s};
    yAll = []; pAll = [];
    for e = 1:numel(elements)
        el = elements{e};
        te = strcmp(T.a1, el) | strcmp(T.a2, el) | strcmp(T.b1, el) | strcmp(T.b2, el);
        [Xtr, Xte] = standardize_train(X(~te, :), X(te, :));
        yAll = [yAll; y(te)]; %#ok<AGROW>
        pAll = [pAll; fit_predict(2, Xtr, y(~te), Xte, SEED)]; %#ok<AGROW>
    end
    [r2, rmse, mae] = metrics(yAll, pAll);
    fprintf('%-20s R2 %.3f  RMSE %.3f eV  MAE %.3f eV\n', fsNames{s}, r2, rmse, mae);
    loeoRows(end + 1, :) = {fsNames{s}, 'Random forest (TreeBagger)', r2, rmse, mae}; %#ok<SAGROW>
end
writetable(cell2table(loeoRows, 'VariableNames', {'features', 'model', 'R2', 'RMSE_eV', 'MAE_eV'}), ...
    fullfile(outDir, 'loeo_results_matlab.csv'));

%% 6. Figure: parity, baseline RF vs site-resolved RF (out-of-fold)
fig = figure('Units', 'inches', 'Position', [0 0 10 4.6], 'Color', 'white');
titles = {'Baseline features', 'Site-resolved features'};
for s = 1:2
    subplot(1, 2, s);
    scatter(y, oof{s, 2}, 10, [0.18 0.43 0.70], 'filled', 'MarkerFaceAlpha', 0.5);
    hold on; plot([0 9], [0 9], 'k--', 'LineWidth', 1);
    [r2, rmse] = metrics(y, oof{s, 2});
    title(sprintf('%s, TreeBagger\nR^2 = %.3f, RMSE = %.2f eV (pooled OOF)', titles{s}, r2, rmse), ...
        'FontWeight', 'normal', 'FontSize', 10);
    xlabel('GLLB-SC band gap (eV)'); ylabel('Predicted, out-of-fold (eV)');
    xlim([0 9]); ylim([0 9]); axis square; box off;
end
exportgraphics(fig, fullfile(outDir, 'parity_matlab.png'), 'Resolution', 200);
fprintf('\nWrote cv_results_matlab.csv, loeo_results_matlab.csv, parity_matlab.png to output/\n');

%% ── Helpers ─────────────────────────────────────────────────────────────
function [Xtr, Xte] = standardize_train(Xtr, Xte)
    % z-score with TRAINING statistics only (no test-set leakage)
    mu = mean(Xtr, 1);
    sd = std(Xtr, 0, 1);
    sd(sd == 0) = 1;
    Xtr = (Xtr - mu) ./ sd;
    Xte = (Xte - mu) ./ sd;
end

function yhat = fit_predict(model, Xtr, ytr, Xte, seed)
    switch model
        case 1
            yhat = predict(fitlm(Xtr, ytr), Xte);
        case 2
            rng(seed);
            mdl = TreeBagger(200, Xtr, ytr, 'Method', 'regression', 'MinLeafSize', 5, ...
                'NumPredictorsToSample', ceil(size(Xtr, 2) / 3));
            yhat = predict(mdl, Xte);
        case 3
            mdl = fitrsvm(Xtr, ytr, 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', false);
            yhat = predict(mdl, Xte);
    end
end

function [R2, RMSE, MAE] = metrics(yTrue, yPred)
    err = yTrue - yPred;
    R2 = 1 - sum(err .^ 2) / sum((yTrue - mean(yTrue)) .^ 2);
    RMSE = sqrt(mean(err .^ 2));
    MAE = mean(abs(err));
end
