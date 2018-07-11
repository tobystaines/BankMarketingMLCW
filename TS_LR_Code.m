   %% Logitic Regression For Prediction of Long-Term Deposit Takeup %%
% Author - Toby Staines
clearvars;
close all;
clc;
%% Load and Prepare Data

[train,X_train,y_train] = dataPrep('ML_CW_TrainingData.csv');
[val,X_val,y_val] = dataPrep('ML_CW_ValidationData.csv');
[test,X_test,y_test] = dataPrep('ML_CW_TestData.csv');


%% Build Logistic Regression Model

logMdl = fitglm(train, 'Distribution', 'binomial', 'Link', 'logit');

%From the model, caluculate probaility of 'Yes' response for validation set (P(Deposit=yes|Data))
scores_val = predict(logMdl,val);

%Calculate the ROC curves for the taining and validation sets
scores_train = logMdl.Fitted.Probability;
[rocX_train, rocY_train, rocT_train, AUC_train] = perfcurve(y_train,scores_train, 1);
[rocX_val, rocY_val, rocT_val, AUC_val] = perfcurve(y_val,scores_val, 1);

%% Investigate Cost

% False positive (FP) and false negative (FN) predictions each have a cost. 
% Here we invetigate the impact of varying the relative weights of these 
% costs on the model's outputs.

alpha = 1; % FN cost parameter - this will be fixed as we vary beta
values = zeros(30,6);

for beta = (0.5:0.5:15)
    [minCostThreshold,~] = minCost(alpha,beta,y_val,rocX_val,rocY_val,rocT_val);
    
    %Calculate predicted classifications 
    valPrediction = scores_val > minCostThreshold;

    %Produce model measures
    [~,valAccuracy,valError,valSensitivity,valSpecificity,valPrecision] = confusion(y_val,valPrediction);
    
    values(beta*2,:) = [beta, valAccuracy,valError,valSensitivity,valSpecificity,valPrecision];
end

% Plot the results of the above experiment
figure; hold on;

plot(values(:,1),values(:,2))
plot(values(:,1),values(:,3))
plot(values(:,1),values(:,4))
plot(values(:,1),values(:,5))
plot(values(:,1),values(:,6))

xlabel('Relative Cost of False Negative Over False Positive')
ylabel('Measure Rate')
title('Logistic Regression Performance Measures - Change with Varying Cost')
legend('Accuracy','Error','Sensitivity','Specificity','Precision','location','southeast')

%Model cost for maximum total accuracy
[~,B] = max(values(:,2));
betaMxA = values(B,1);
[maxAccThreshold, maxAidx] = minCost(alpha,betaMxA,y_val,rocX_val,rocY_val,rocT_val);

% The above produces a the best overall accuracy. However, because the data
% is heavily skewed toward negative results, the model greatly favours
% specificity over sensitivity. In the context of a bank's marketing
% campaign it is likely that they would view FN (missed sales opportunity 
% as having a higher cost than FP (wasted phone call). Based on this
% intuition and the results of the cost experiment above we also run the
% model using the threshold which produces a sensitivity in the validation
% data of at least 70%

%Model cost with a minimum sensitivity of 70%
alpha = 1;
betaMC = values(find((values(:,4)>0.70),1),1);
[minCostThreshold, minCidx] = minCost(alpha,betaMC,y_val,rocX_val,rocY_val,rocT_val);



%Calculate predicted classifications for training and validation data sets 
%based on the minimum cost and maximum accuracy thresholds 
trainMCPrediction = scores_train > minCostThreshold;
valMCPrediction = scores_val > minCostThreshold;
trainMxAPrediction = scores_train > maxAccThreshold;
valMxAPrediction = scores_val > maxAccThreshold;

%Produce a confusion Matrix of the resulting predictions
[trainMCConfMat,trainMCAccuracy,trainMCError,trainMCSensitivity,trainMCSpecificity,trainMCPrecision] = confusion(y_train,trainMCPrediction);
[valMCConfMat,valMCAccuracy,valMCError,valMCSensitivity,valMCSpecificity,valMCPrecision] = confusion(y_val,valMCPrediction);
[trainMxAConfMat,trainMxAAccuracy,trainMxAError,trainMxASensitivity,trainMxASpecificity,trainMxAPrecision] = confusion(y_train,trainMxAPrediction);
[valMxAConfMat,valMxAAccuracy,valMxAError,valMxASensitivity,valMxASpecificity,valMxAPrecision] = confusion(y_val,valMxAPrediction);

%% Run Model on Test Data

scores_test = predict(logMdl,test);

%Calculate predicted classifications based on the minimum cost threshold
testMCPrediction = scores_test > minCostThreshold;
testMxAPrediction = scores_test > maxAccThreshold;

% Calculte ROC for test data
[rocX_test, rocY_test, rocT_test, AUC_test] = perfcurve(y_test,scores_test, 1);

%Produce a confusion Matrix of the resulting predictions
[testMCConfMat,testMCAccuracy,testMCError,testMCSensitivity,testMCSpecificity,testMCPrecision] = confusion(y_test,testMCPrediction);
[testMxAConfMat,testMxAAccuracy,testMxAError,testMxASensitivity,testMxASpecificity,testMxAPrecision] = confusion(y_test,testMxAPrediction);

%% Plot ROC Curves
figure; hold on; axis([0 1 0 1]);


% Plot ROC curves for Logistic Regression
plot(rocX_train,rocY_train)
plot(rocX_val, rocY_val)
plot(rocX_test, rocY_test)
plot(rocX_val(minCidx),rocY_val(minCidx),'o','markersize',12)
plot(rocX_val(maxAidx),rocY_val(maxAidx),'ro','markersize',12)


% Plot ROC curves for Naive Bayes
NBTrainROC = readtable('NBTrainROC.csv');
NBValROC = readtable('NBValROC.csv');
NBTestROC = readtable('NBTestROC.csv');
NBTrainX = table2array(NBTrainROC(:,1));
NBTrainY = table2array(NBTrainROC(:,2));
NBValX = table2array(NBValROC(:,1));
NBValY = table2array(NBValROC(:,2));
NBTestX = table2array(NBTestROC(:,1));
NBTestY = table2array(NBTestROC(:,2));
plot(NBTrainX,NBTrainY)
plot(NBValX,NBValY)
plot(NBTestX,NBTestY)

% Plot random guess line
randomClassifierLineX= linspace(0,1)';
randomClassifierLineY= linspace(0,1)';
plot(randomClassifierLineX, randomClassifierLineY, 'k--')

xlabel('False positive rate')
ylabel('True positive rate')
rocTitle = 'ROC Comparison:';
rocTitle = [rocTitle newline 'Logistic Regression vs Naive Bayes'];
title(rocTitle)
legend('LR Training Performance','LR Validation Performance',... 
       'LR Test Performance','LR 70% Sensitivity Threshold',... 
       'LR Maximum Accuracy Threshold','NB Training Performance',...
       'NB Validation Performance','NB Test Performance',...
       'Random guess Performance','location','southeast')


%% Logistic Regression Outputs

% LR Model Coefficients
logMdl.Coefficients

% AUC for Training and Validation sets
AUC_train
AUC_val
AUC_test

% The classification thresholds determined for maximum accuracy and for
% desired sensitivity
maxAccThreshold
minCostThreshold

% Results Measures
lrResults = zeros(6,5);
lrResults(1,:) = [trainMxAAccuracy,trainMxAError,trainMxASensitivity,trainMxASpecificity,trainMxAPrecision];
lrResults(2,:) = [valMxAAccuracy,valMxAError,valMxASensitivity,valMxASpecificity,valMxAPrecision];
lrResults(3,:) = [testMxAAccuracy,testMxAError,testMxASensitivity,testMxASpecificity,testMxAPrecision];
lrResults(4,:) = [trainMCAccuracy,trainMCError,trainMCSensitivity,trainMCSpecificity,trainMCPrecision];
lrResults(5,:) = [valMCAccuracy,valMCError,valMCSensitivity,valMCSpecificity,valMCPrecision];
lrResults(6,:) = [testMCAccuracy,testMCError,testMCSensitivity,testMCSpecificity,testMCPrecision];
lrResults = array2table(lrResults,'VariableNames',{'Accuracy','Error','Sensitivity','Specificity','Precision'},...
    'RowNames',{'Training - Maximum Accuracy','Validation - Maximum Accuracy','Test - Maximum Accuracy',...
    'Training - 70% Sensitivity','Validation - 70% Sensitivity','Test - 70% Sensitivity'})

% Confusion Matrixes - Maximum Accuracy
trainMxAConfMat
valMxAConfMat
testMxAConfMat
% Confusion Matrixes - 70% Min Sensitivity on Validation Data
trainMCConfMat
valMCConfMat
testMCConfMat
