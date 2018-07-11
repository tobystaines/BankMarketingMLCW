% THIS CODE WAS USED TO SPLIT THE ORIGINAL DATA SET IN TO TRAIN/VAL/TEST.
% IF IT RUN AGAIN THE SAMPLING WILL BE DIFFERENT, LEADING TO DIFFERENT
% RESULTS FROM THOSE DESCRIBED IN THE POSTER
%Author - Toby Staines
clear;
clc;
%% Load and Prep Data

%Load UCI Bank Marketing Data
data = readtable('bank-additional-full.csv');

%Split data in to training, validation, and test dets
[trainIdx, valIdx, testIdx] = dividerand(height(data),0.7,0.15,0.15);
trainD = data(trainIdx,:);
valD = data(valIdx,:);
testD = data(testIdx,:);

%Output separate data sets to new csv files
writetable(trainD,'ML_CW_TrainingData.csv');
writetable(valD,'ML_CW_ValidationData.csv');
writetable(testD,'ML_CW_TestData.csv');
