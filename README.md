# BankMarketingMLCW
Selecting target customers with logistic regression in MATLAB on the UCI Bank Marketing dataset

Code should be run from the same folder as the data or it will not work.

Data is available at: https://archive.ics.uci.edu/ml/datasets/bank+marketing

1. Logistic Regression Code:
	
TS_LR_Code.m - This is the main code for the logistic regression model, which should be run to re-produce
	   our LR results.
	
dataPrep.m - A function called by TS_LR_Code to preprocess each of the data sets before modelling
	
minCost.m - A cost function called by TS_LR_Code
	
confusion.m - A function called by TS_LR_Code to produce a confusion matrix and other model performance measures

2. Other Code:
	
CourseWorkDataPrepCode.m - This takes the original data file and splits it in to training, 
	   validation and test data. 
