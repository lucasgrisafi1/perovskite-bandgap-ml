%% Materials Informatics: Perovskite Band Gap Predictor
%  Predicts GLLB-SC band gaps of 1306 double perovskites (Pilania et al.,
%  Sci. Rep. 2016, "Machine learning bandgaps of double perovskites";
%  matminer dataset 'double_perovskites_gap') from 5 compositional
%  descriptors. Trains and compares fitlm, TreeBagger, and fitrsvm.
%
%  Requires: Statistics and Machine Learning Toolbox
%  Run from the matlab/ folder. Outputs go to ../output/.
%
%  Lucas Grisafi | GT MSE | Summer 2026 Project 2

clear; clc; close all;

%% ── 1. Load and inspect data ───────────────────────────────────────────
T = readtable(fullfile('..', 'data', 'double_perovskites_gap.csv'));

disp(head(T, 5));
disp(T.Properties.VariableNames);

missing_counts = sum(ismissing(T));
disp(table(T.Properties.VariableNames', missing_counts', ...
    'VariableNames', {'Column', 'MissingCount'}));

% Keep only rows with valid band gap (also filters metals, gap = 0)
T = T(T.gap_gllbsc > 0, :);
fprintf('Compounds after filtering: %d\n', height(T));

%% ── 2. Feature engineering — compositional descriptors ────────────────
% Hypothesis: band gap depends on orbital overlap, which is governed by
% electronegativity differences, atomic size, and valence electron count.
props = element_props();
feature_names = {'Mean EN', 'Std EN', 'Mean Radius', 'Mean Valence', 'Radius Ratio'};

n_compounds = height(T);
n_features  = 5;
X = zeros(n_compounds, n_features);
y = T.gap_gllbsc;

for i = 1:n_compounds
    formula = T.formula{i};                      % e.g. 'AgNbLaAlO6'
    [elements, counts] = parse_formula(formula);

    EN      = zeros(1, numel(elements));
    radius  = zeros(1, numel(elements));
    valence = zeros(1, numel(elements));
    for j = 1:numel(elements)
        ep = props.(elements{j});
        EN(j)      = ep(2);
        radius(j)  = ep(3);
        valence(j) = ep(4);
    end

    w = counts / sum(counts);                    % stoichiometric weights
    X(i,1) = sum(w .* EN);                       % mean electronegativity
    X(i,2) = std(EN, 1);                         % EN diversity (population std)
    X(i,3) = sum(w .* radius);                   % mean atomic radius
    X(i,4) = sum(w .* valence);                  % mean valence electrons
    X(i,5) = max(radius) / min(radius);          % radius ratio
end

%% ── 3. Clean, normalize, split ─────────────────────────────────────────
valid = all(~isnan(X), 2) & ~isnan(y);
X = X(valid, :);  y = y(valid);
fprintf('Final dataset: %d compounds\n', length(y));

[X_norm, mu, sigma] = zscore(X);   % save mu/sigma to score new compounds

rng(42);                           % reproducible split
cv = cvpartition(length(y), 'HoldOut', 0.2);
X_tr = X_norm(cv.training, :);  y_tr = y(cv.training);
X_te = X_norm(cv.test,     :);  y_te = y(cv.test);
fprintf('Train: %d  Test: %d\n', sum(cv.training), sum(cv.test));

%% ── 4. Train three models ──────────────────────────────────────────────
% Model 1: Linear Regression (baseline)
mdl_lm  = fitlm(X_tr, y_tr);
yhat_lm = predict(mdl_lm, X_te);
[R2_lm, RMSE_lm] = compute_metrics(y_te, yhat_lm);

% Model 2: Random Forest
mdl_rf = TreeBagger(100, X_tr, y_tr, ...
    'Method',                 'regression', ...
    'OOBPrediction',          'on', ...
    'OOBPredictorImportance', 'on', ...
    'MinLeafSize',            5);
yhat_rf = predict(mdl_rf, X_te);   % regression TreeBagger returns doubles
[R2_rf, RMSE_rf] = compute_metrics(y_te, yhat_rf);

% Model 3: Support Vector Regression (RBF kernel)
mdl_svm  = fitrsvm(X_tr, y_tr, 'KernelFunction', 'rbf', 'Standardize', false);
yhat_svm = predict(mdl_svm, X_te);
[R2_svm, RMSE_svm] = compute_metrics(y_te, yhat_svm);

% Comparison table
fprintf('\n%-20s %8s %8s\n', 'Model', 'R2', 'RMSE');
fprintf('%-20s %8.3f %8.3f\n', 'Linear Regression', R2_lm,  RMSE_lm);
fprintf('%-20s %8.3f %8.3f\n', 'Random Forest',     R2_rf,  RMSE_rf);
fprintf('%-20s %8.3f %8.3f\n', 'SVM (RBF)',         R2_svm, RMSE_svm);

comparison = table( ...
    {'Linear Regression'; 'Random Forest'; 'SVM (RBF)'}, ...
    [R2_lm; R2_rf; R2_svm], [RMSE_lm; RMSE_rf; RMSE_svm], ...
    'VariableNames', {'Model', 'R2', 'RMSE_eV'});
if ~exist(fullfile('..', 'output'), 'dir'), mkdir(fullfile('..', 'output')); end
writetable(comparison, fullfile('..', 'output', 'model_comparison_matlab.csv'));

%% ── 5. Figure 1: Predicted vs actual (3 models) ───────────────────────
fig1 = figure('Units', 'inches', 'Position', [0 0 12 4], 'Color', 'white');
models     = {yhat_lm, yhat_rf, yhat_svm};
modelnames = {'Linear Regression', 'Random Forest', 'SVM (RBF)'};
R2s        = [R2_lm, R2_rf, R2_svm];
colors     = {[0.2 0.5 0.8], [0.1 0.6 0.3], [0.8 0.3 0.2]};

for m = 1:3
    subplot(1, 3, m);
    scatter(y_te, models{m}, 20, colors{m}, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    lims = [min(y_te)*0.9, max(y_te)*1.1];
    plot(lims, lims, 'k--', 'LineWidth', 1);
    xlabel('Actual Band Gap (eV)', 'FontSize', 10);
    ylabel('Predicted Band Gap (eV)', 'FontSize', 10);
    title(sprintf('%s\nR^2 = %.3f', modelnames{m}, R2s(m)), ...
        'FontWeight', 'normal', 'FontSize', 10);
    xlim(lims); ylim(lims); axis square; box off;
end
exportgraphics(fig1, fullfile('..', 'output', 'predicted_vs_actual.png'), 'Resolution', 300);

%% ── 6. Figure 2: Feature importance (Random Forest, OOB permutation) ──
importance = mdl_rf.OOBPermutedPredictorDeltaError;
[sorted_imp, idx] = sort(importance, 'descend');

fig2 = figure('Units', 'inches', 'Position', [0 0 6 4], 'Color', 'white');
barh(flip(sorted_imp), 'FaceColor', [0.2 0.6 0.4]);
set(gca, 'YTickLabel', flip(feature_names(idx)), 'Box', 'off', 'TickDir', 'out');
xlabel('Increase in OOB MSE when permuted', 'FontSize', 10);
title('Feature Importance — Random Forest', 'FontWeight', 'normal');
exportgraphics(fig2, fullfile('..', 'output', 'feature_importance.png'), 'Resolution', 300);

fprintf('\nFigures and comparison table saved to output/\n');

%% ── Helper ─────────────────────────────────────────────────────────────
function [R2, RMSE] = compute_metrics(y_true, y_pred)
    SS_res = sum((y_true - y_pred).^2);
    SS_tot = sum((y_true - mean(y_true)).^2);
    R2   = 1 - SS_res / SS_tot;
    RMSE = sqrt(mean((y_true - y_pred).^2));
end
