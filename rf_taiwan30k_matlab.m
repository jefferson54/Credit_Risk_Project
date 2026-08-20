%% rf_taiwan30k_matlab.m
% Taiwan 30k (X1..X23) -> Target Y, Random Forest (TreeBagger) + AUC/F1 + permutation importance
% Works on older MATLAB (no oobIndices / no oobPermutedPredictorImportance needed)

clear; clc; close all;
rng(42);

FILE   = "taiwan_30k.csv";
TARGET = "Y";

%% Load (preserve headers exactly)
T = readtable(FILE, "VariableNamingRule","preserve");

% Drop index-like column if present
if any(strcmp(T.Properties.VariableNames, "Unnamed: 0"))
    T = removevars(T, "Unnamed: 0");
end

% Assert target
assert(any(strcmp(T.Properties.VariableNames, TARGET)), "Target column not found.");

y = T.(TARGET);
X = removevars(T, TARGET);

% Ensure numeric 0/1 target
y = double(y);
y = (y ~= 0);

% Predictors matrix
Xm = table2array(X);
featNames = string(X.Properties.VariableNames);

%% Split 70/15/15 stratified (train/test used; val split kept but not required)
cv1 = cvpartition(y, "HoldOut", 0.30);
idxTrain = training(cv1);
idxTemp  = test(cv1);

yTemp = y(idxTemp);
cv2 = cvpartition(yTemp, "HoldOut", 0.50);

tempIdx = find(idxTemp);
idxVal  = false(size(y)); idxVal(tempIdx(training(cv2))) = true; %#ok<NASGU>
idxTest = false(size(y)); idxTest(tempIdx(test(cv2)))    = true;

Xtr = Xm(idxTrain,:); ytr = y(idxTrain);
Xte = Xm(idxTest,:);  yte = y(idxTest);

%% Train Random Forest
nTrees  = 400;
minLeaf = 10;
mtry    = max(1, round(sqrt(size(Xtr,2))));

Mdl = TreeBagger( ...
    nTrees, Xtr, ytr, ...
    "Method","classification", ...
    "OOBPrediction","on", ...
    "MinLeafSize",minLeaf, ...
    "NumPredictorsToSample",mtry);

%% Predict (probability of class 1)
[~, p_te] = predict(Mdl, Xte);
p1 = p_te(:,2);
yhat = p1 >= 0.5;

%% Metrics
TP = sum((yhat==1) & (yte==1));
TN = sum((yhat==0) & (yte==0));
FP = sum((yhat==1) & (yte==0));
FN = sum((yhat==0) & (yte==1));

acc  = (TP+TN) / (TP+TN+FP+FN);
prec = TP / max(1,(TP+FP));
rec  = TP / max(1,(TP+FN));
f1   = 2*prec*rec / max(1e-12,(prec+rec));

[Xroc, Yroc, ~, AUC] = perfcurve(double(yte), p1, 1);

fprintf("\n=== TEST METRICS (MATLAB / Taiwan 30k) ===\n");
fprintf("Accuracy : %.4f\n", acc);
fprintf("Precision: %.4f\n", prec);
fprintf("Recall   : %.4f\n", rec);
fprintf("F1       : %.4f\n", f1);
fprintf("AUC      : %.4f\n\n", AUC);

%% Plots
figure; plot(Xroc, Yroc); grid on;
xlabel("False Positive Rate"); ylabel("True Positive Rate");
title(sprintf("ROC Curve (AUC = %.4f)", AUC));

figure; cm = confusionmat(double(yte), double(yhat));
confusionchart(cm, ["0","1"]);
title("Confusion Matrix (Test)");

%% Feature importance (Permutation on TEST via AUC drop — version-safe)
baselineAUC = AUC;
nFeat = numel(featNames);
imp_auc_drop = zeros(nFeat,1);

for j = 1:nFeat
    Xte_perm = Xte;
    Xte_perm(:,j) = Xte_perm(randperm(size(Xte_perm,1)), j);

    [~, p_perm] = predict(Mdl, Xte_perm);
    p1_perm = p_perm(:,2);

    [~,~,~, AUC_perm] = perfcurve(double(yte), p1_perm, 1);
    imp_auc_drop(j) = baselineAUC - AUC_perm;   % higher = more important
end

[impSorted, idx] = sort(imp_auc_drop, "descend");
topK = min(20, nFeat);

figure;
bar(impSorted(1:topK)); grid on;
xticks(1:topK); xticklabels(featNames(idx(1:topK)));
xtickangle(45);
ylabel("AUC Drop After Permutation");
title("Top Feature Importances (Permutation Test AUC Drop)");

% Save importance + model
writetable(table(featNames', imp_auc_drop, 'VariableNames', ["feature","importance_auc_drop"]), ...
    "feature_importance_matlab.csv");

save("rf_taiwan30k_matlab_model.mat", "Mdl", "featNames", "TARGET", ...
     "acc","prec","rec","f1","AUC");
