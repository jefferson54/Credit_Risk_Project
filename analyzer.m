%% analyzer.m
% Inspect Taiwan 30k CSV structure and column names

clear; clc;

FILE = "taiwan_30k.csv";   % CSV must be in the same folder

% Read table while preserving original column headers
T = readtable(FILE, "VariableNamingRule","preserve");

% Basic info
disp("=== DATASET SIZE ===");
disp(size(T));

disp(" ");
disp("=== COLUMN NAMES (EXACT) ===");
disp(string(T.Properties.VariableNames)');

disp(" ");
disp("=== FIRST 5 ROWS ===");
disp(T(1:5, :));
