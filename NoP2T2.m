clc;
clear;
close all;

%% ==========================
% Step 1: Read Data
% ==========================
data = readmatrix('DATA (2).xlsx');

Pressure = data(:,1);
Temperature = data(:,2);
DeltaI_measured = round(data(:,3), 2);
DeltaI_ref      = round(data(:,4), 2);

% Construct residual
y_residual = DeltaI_measured - DeltaI_ref;

%% ==========================
% Step 2: Construct Input Features (P, T, P×T)
% ==========================
X = [Pressure, Temperature, Pressure .* Temperature];

%% ==========================
% Step 3: Split Dataset (70% Train / 20% Test / 10% Valid)
% ==========================
n = size(X,1);
idx = randperm(n);

n_train = round(0.7 * n);
n_test  = round(0.2 * n);
n_valid = n - n_train - n_test;

idx_train = idx(1:n_train);
idx_test  = idx(n_train+1:n_train+n_test);
idx_valid = idx(n_train+n_test+1:end);

X_train = X(idx_train,:);   y_train = y_residual(idx_train);
X_test  = X(idx_test,:);    y_test  = y_residual(idx_test);
X_valid = X(idx_valid,:);   y_valid = y_residual(idx_valid);

y_ref_train  = DeltaI_ref(idx_train);
y_ref_test   = DeltaI_ref(idx_test);
y_ref_valid  = DeltaI_ref(idx_valid);

y_meas_train = DeltaI_measured(idx_train);
y_meas_test  = DeltaI_measured(idx_test);
y_meas_valid = DeltaI_measured(idx_valid);

%% ==========================
% Step 4: Train DNN Model
% ==========================
net = fitrnet(X_train, y_train, ...
    'Standardize', true, ...
    'LayerSizes', [64 32 16], ...
    'Activations', 'relu', ...
    'Lambda', 1e-4, ...
    'IterationLimit', 2000);

%% ==========================
% Step 5: Predict and Decouple
% ==========================
y_train_pred = y_meas_train - predict(net, X_train);
y_test_pred  = y_meas_test  - predict(net, X_test);
y_valid_pred = y_meas_valid - predict(net, X_valid);

%% ==========================
% Step 6: Define Error Metrics
% ==========================
computeMetrics = @(y_pred, y_ref) struct( ...
    'MAE', mean(abs(y_pred - y_ref)), ...
    'MSE', mean((y_pred - y_ref).^2), ...
    'R2', 1 - sum((y_pred - y_ref).^2) / sum((y_ref - mean(y_ref)).^2), ...
    'MaxRelErr', max(abs(y_pred - y_ref) ./ max(abs(y_ref),1e-6)) ...
);

% Before decoupling
m_train_raw = computeMetrics(y_meas_train, y_ref_train);
m_test_raw  = computeMetrics(y_meas_test,  y_ref_test);
m_valid_raw = computeMetrics(y_meas_valid, y_ref_valid);

% After decoupling
m_train_dnn = computeMetrics(y_train_pred, y_ref_train);
m_test_dnn  = computeMetrics(y_test_pred,  y_ref_test);
m_valid_dnn = computeMetrics(y_valid_pred, y_ref_valid);

%% ==========================
% Step 7: Plot Regression Curves
% ==========================
figure('Name','Regression Comparison (Raw vs DNN)','Color','w');

subplot(1,3,1);
plot(y_ref_train, y_meas_train, 'ro'); hold on;
plot(y_ref_train, y_train_pred, 'go');
plot([min(y_ref_train), max(y_ref_train)], ...
     [min(y_ref_train), max(y_ref_train)], 'k--');
title('Train Set'); xlabel('Reference'); ylabel('Measured / Predicted');
legend('Raw','DNN','Location','best');

subplot(1,3,2);
plot(y_ref_test, y_meas_test, 'ro'); hold on;
plot(y_ref_test, y_test_pred, 'go');
plot([min(y_ref_test), max(y_ref_test)], ...
     [min(y_ref_test), max(y_ref_test)], 'k--');
title('Test Set'); xlabel('Reference'); ylabel('Measured / Predicted');
legend('Raw','DNN','Location','best');

subplot(1,3,3);
plot(y_ref_valid, y_meas_valid, 'ro'); hold on;
plot(y_ref_valid, y_valid_pred, 'go');
plot([min(y_ref_valid), max(y_ref_valid)], ...
     [min(y_ref_valid), max(y_ref_valid)], 'k--');
title('Validation Set'); xlabel('Reference'); ylabel('Measured / Predicted');
legend('Raw','DNN','Location','best');

saveas(gcf, 'Regression_Comparison_NoP2T2.png');

%% ==========================
% Step 8: Save Metrics to Excel
% ==========================
T_metrics = table( ...
    ["MAE";"MSE";"R2";"MaxRelErr"], ...
    [m_train_raw.MAE; m_train_raw.MSE; m_train_raw.R2; m_train_raw.MaxRelErr], ...
    [m_train_dnn.MAE; m_train_dnn.MSE; m_train_dnn.R2; m_train_dnn.MaxRelErr], ...
    [m_test_raw.MAE;  m_test_raw.MSE;  m_test_raw.R2;  m_test_raw.MaxRelErr], ...
    [m_test_dnn.MAE;  m_test_dnn.MSE;  m_test_dnn.R2;  m_test_dnn.MaxRelErr], ...
    [m_valid_raw.MAE; m_valid_raw.MSE; m_valid_raw.R2; m_valid_raw.MaxRelErr], ...
    [m_valid_dnn.MAE; m_valid_dnn.MSE; m_valid_dnn.R2; m_valid_dnn.MaxRelErr], ...
    'VariableNames', {'Metric','Train_Before','Train_After','Test_Before','Test_After','Valid_Before','Valid_After'} ...
);

writetable(T_metrics, 'Decoupling_Metrics_NoP2T2.xlsx', ...
           'Sheet', 'Error_Comparison');

disp('=== Error Comparison (Before vs After Decoupling) ===');
disp(T_metrics);

fprintf("✅ Done. Results saved to Excel and regression figure exported.\n");
