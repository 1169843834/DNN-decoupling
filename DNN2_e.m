clc;
clear;
close all;

%% Step 1: Read data
data = readmatrix('DATA (2).xlsx');
Pressure = data(:,1);
Temperature = data(:,2);
DeltaI_measured = round(data(:,3), 2);
DeltaI_ref      = round(data(:,4), 2);

% Construct input features and residuals
X = [Pressure, Temperature];
y_residual = DeltaI_measured - DeltaI_ref;

% Augment features with nonlinear terms
X_aug = [Pressure, Temperature, Pressure.^2, Temperature.^2, Pressure.*Temperature];

%% Step 2: Split dataset (70% train, 20% test, 10% validation)
%%rng(10); % Optional: fix random seed
n = size(X_aug,1);
idx = randperm(n);
n_train = round(0.7 * n);
n_test  = round(0.2 * n);
n_valid = n - n_train - n_test;

idx_train = idx(1:n_train);
idx_test  = idx(n_train+1:n_train+n_test);
idx_valid = idx(n_train+n_test+1:end);

X_train = X_aug(idx_train,:); y_train = y_residual(idx_train);
X_test  = X_aug(idx_test,:);  y_test  = y_residual(idx_test);
X_valid = X_aug(idx_valid,:); y_valid = y_residual(idx_valid);

y_ref_train = DeltaI_ref(idx_train);
y_ref_test  = DeltaI_ref(idx_test);
y_ref_valid = DeltaI_ref(idx_valid);

y_meas_train = DeltaI_measured(idx_train);
y_meas_test  = DeltaI_measured(idx_test);
y_meas_valid = DeltaI_measured(idx_valid);

%% Step 3: Train DNN model (fitrnet) with timing
tic;   % ----------- Start timing -----------


net = fitrnet(X_train, y_train, ...
    'Standardize', true, ...
    'LayerSizes', [64 32 16], ...
    'Activations', 'relu', ...
    'Lambda', 1e-4, ...
    'IterationLimit', 2000);

train_time_ms = toc * 1000;   % convert seconds → milliseconds

fprintf("⏱️ DNN training time: %.2f ms\n", train_time_ms);


%% Step 4: Predict residuals and perform decoupling
y_train_pred = y_meas_train - predict(net, X_train);
y_test_pred  = y_meas_test  - predict(net, X_test);
y_valid_pred = y_meas_valid - predict(net, X_valid);
%% Measure single-sample inference time
sample = X_test(1,:);  % take one sample

t_inf = tic;
pred_single = predict(net, sample);
infer_time_ms = toc(t_inf) * 1000;    % convert to ms
infer_time_us = infer_time_ms * 1000; % convert to microseconds

fprintf("🚀 Single-sample inference time: %.4f ms (%.2f μs)\n", ...
    infer_time_ms, infer_time_us);


%% Step 5: Define error metrics function
computeMetrics = @(y_pred, y_ref) struct( ...
    'MAE', mean(abs(y_pred - y_ref)), ...
    'MSE', mean((y_pred - y_ref).^2), ...
    'R2', 1 - sum((y_pred - y_ref).^2)/sum((y_ref - mean(y_ref)).^2), ...
    'MaxRelErr', max(abs(y_pred - y_ref)./abs(y_ref)));

% Before decoupling
metrics_train_raw = computeMetrics(y_meas_train, y_ref_train);
metrics_test_raw  = computeMetrics(y_meas_test,  y_ref_test);
metrics_valid_raw = computeMetrics(y_meas_valid, y_ref_valid);

% After decoupling
metrics_train_dnn = computeMetrics(y_train_pred, y_ref_train);
metrics_test_dnn  = computeMetrics(y_test_pred,  y_ref_test);
metrics_valid_dnn = computeMetrics(y_valid_pred, y_ref_valid);

%% Step 6: Plot regression curves for each subset
figure('Name','Regression Comparison','Color','w'); hold on;
subplot(1,3,1);
plot(y_ref_train, y_meas_train, 'ro'); hold on;
plot(y_ref_train, y_train_pred, 'go'); 
plot([min(y_ref_train),max(y_ref_train)], [min(y_ref_train),max(y_ref_train)], 'k--');
title('Train Set'); xlabel('Reference'); ylabel('Measured / Predicted'); legend('Raw','DNN');

subplot(1,3,2);
plot(y_ref_test, y_meas_test, 'ro'); hold on;
plot(y_ref_test, y_test_pred, 'go'); 
plot([min(y_ref_test),max(y_ref_test)], [min(y_ref_test),max(y_ref_test)], 'k--');
title('Test Set'); xlabel('Reference'); ylabel('Measured / Predicted'); legend('Raw','DNN');

subplot(1,3,3);
plot(y_ref_valid, y_meas_valid, 'ro'); hold on;
plot(y_ref_valid, y_valid_pred, 'go'); 
plot([min(y_ref_valid),max(y_ref_valid)], [min(y_ref_valid),max(y_ref_valid)], 'k--');
title('Validation Set'); xlabel('Reference'); ylabel('Measured / Predicted'); legend('Raw','DNN');

saveas(gcf, 'Regression_Comparison.png');

%% Step 7: Save metrics to Excel
T_metrics = table( ...
    ["MAE";"MSE";"R2";"MaxRelErr"], ...
    [metrics_train_raw.MAE;metrics_train_raw.MSE;metrics_train_raw.R2;metrics_train_raw.MaxRelErr], ...
    [metrics_train_dnn.MAE;metrics_train_dnn.MSE;metrics_train_dnn.R2;metrics_train_dnn.MaxRelErr], ...
    [metrics_test_raw.MAE;metrics_test_raw.MSE;metrics_test_raw.R2;metrics_test_raw.MaxRelErr], ...
    [metrics_test_dnn.MAE;metrics_test_dnn.MSE;metrics_test_dnn.R2;metrics_test_dnn.MaxRelErr], ...
    [metrics_valid_raw.MAE;metrics_valid_raw.MSE;metrics_valid_raw.R2;metrics_valid_raw.MaxRelErr], ...
    [metrics_valid_dnn.MAE;metrics_valid_dnn.MSE;metrics_valid_dnn.R2;metrics_valid_dnn.MaxRelErr], ...
    'VariableNames', {'Metric','Train_Before','Train_After','Test_Before','Test_After','Valid_Before','Valid_After'});

writetable(T_metrics, 'Decoupling_Metrics_Results.xlsx', 'Sheet', 'Error_Comparison');
disp('=== Error Comparison Across Subsets (Before vs After Decoupling) ===');
disp(T_metrics);

fprintf("✅ Decoupling completed. Results saved as 'Decoupling_Metrics_Results.xlsx', figure as 'Regression_Comparison.png'\n");
