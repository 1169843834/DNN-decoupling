clc;
clear;
close all;

%% ===================== Step 1: Read data ==========================
data = readmatrix('DATA (2).xlsx');
Pressure    = data(:,1);
Temperature = data(:,2);
DeltaI_measured = round(data(:,3), 2);
DeltaI_ref      = round(data(:,4), 2);

% Residual target
y_residual = DeltaI_measured - DeltaI_ref;

% ----- Construct shallow model feature set: [P, T, P^2, T^2, P*T] -----
X_aug = [Pressure, Temperature, Pressure.^2, Temperature.^2, Pressure.*Temperature];

%% ===================== Step 2: Split dataset ========================
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

%% ===================== Step 3: Train SHALLOW DNN ====================
% ---- Single hidden layer (e.g., 32 neurons) ----
net_shallow = fitrnet(X_train, y_train, ...
    'Standardize', true, ...
    'LayerSizes', 32, ...       % Only one hidden layer !!!
    'Activations', 'relu', ...
    'Lambda', 1e-4, ...
    'IterationLimit', 2000);

%% ================= Step 4: Predict & Decoupling =====================
y_train_pred = y_meas_train - predict(net_shallow, X_train);
y_test_pred  = y_meas_test  - predict(net_shallow, X_test);
y_valid_pred = y_meas_valid - predict(net_shallow, X_valid);

%% ================= Step 5: Metrics ================================
computeMetrics = @(y_pred, y_ref) struct( ...
    'MAE', mean(abs(y_pred - y_ref)), ...
    'MSE', mean((y_pred - y_ref).^2), ...
    'R2', 1 - sum((y_pred - y_ref).^2)/sum((y_ref - mean(y_ref)).^2), ...
    'MaxRelErr', max(abs(y_pred - y_ref)./abs(y_ref)));

% Before decoupling
metrics_train_raw = computeMetrics(y_meas_train, y_ref_train);
metrics_test_raw  = computeMetrics(y_meas_test,  y_ref_test);
metrics_valid_raw = computeMetrics(y_meas_valid, y_ref_valid);

% After shallow model decoupling
metrics_train_shallow = computeMetrics(y_train_pred, y_ref_train);
metrics_test_shallow  = computeMetrics(y_test_pred,  y_ref_test);
metrics_valid_shallow = computeMetrics(y_valid_pred, y_ref_valid);

%% ================= Step 6: Plot regression ==========================
figure('Name','Shallow DNN Regression','Color','w'); 
subplot(1,3,1);
plot(y_ref_train, y_meas_train, 'ro'); hold on;
plot(y_ref_train, y_train_pred, 'bo');
plot([min(y_ref_train),max(y_ref_train)], [min(y_ref_train),max(y_ref_train)], 'k--');
title('Train Set'); xlabel('Reference'); ylabel('Measured / Predicted');
legend('Raw','Shallow DNN');

subplot(1,3,2);
plot(y_ref_test, y_meas_test, 'ro'); hold on;
plot(y_ref_test, y_test_pred, 'bo');
plot([min(y_ref_test),max(y_ref_test)], [min(y_ref_test),max(y_ref_test)], 'k--');
title('Test Set'); xlabel('Reference'); ylabel('Measured / Predicted');
legend('Raw','Shallow DNN');

subplot(1,3,3);
plot(y_ref_valid, y_meas_valid, 'ro'); hold on;
plot(y_ref_valid, y_valid_pred, 'bo');
plot([min(y_ref_valid),max(y_ref_valid)], [min(y_ref_valid),max(y_ref_valid)], 'k--');
title('Validation Set'); xlabel('Reference'); ylabel('Measured / Predicted');
legend('Raw','Shallow DNN');

saveas(gcf, 'Shallow_Regression.png');

%% ================= Step 7: Save Metrics ============================
T_metrics = table( ...
    ["MAE";"MSE";"R2";"MaxRelErr"], ...
    [metrics_train_raw.MAE; metrics_train_raw.MSE; metrics_train_raw.R2; metrics_train_raw.MaxRelErr], ...
    [metrics_train_shallow.MAE; metrics_train_shallow.MSE; metrics_train_shallow.R2; metrics_train_shallow.MaxRelErr], ...
    [metrics_test_raw.MAE; metrics_test_raw.MSE; metrics_test_raw.R2; metrics_test_raw.MaxRelErr], ...
    [metrics_test_shallow.MAE; metrics_test_shallow.MSE; metrics_test_shallow.R2; metrics_test_shallow.MaxRelErr], ...
    [metrics_valid_raw.MAE; metrics_valid_raw.MSE; metrics_valid_raw.R2; metrics_valid_raw.MaxRelErr], ...
    [metrics_valid_shallow.MAE; metrics_valid_shallow.MSE; metrics_valid_shallow.R2; metrics_valid_shallow.MaxRelErr], ...
    'VariableNames', {'Metric','Train_Before','Train_After','Test_Before','Test_After','Valid_Before','Valid_After'} );

writetable(T_metrics, 'Shallow_DNN_Metrics.xlsx', 'Sheet','Error_Comparison');

disp('=== Shallow Network Evaluation Completed ===');
disp(T_metrics);
