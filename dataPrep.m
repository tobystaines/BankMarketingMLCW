function [data,X,y] = dataPrep(filename)
%Author - Toby Staines

%Load UCI Bank Marketing Data and prepare it for use

%Load the data
data = readtable(filename);
dataOrig = data;

%Set categorical data types and discretise continuous variables
data.AgeBin = categorical(discretize(data.age,5));
edges=[0 1 56];
data.CampaignBin = categorical(discretize(data.campaign,edges','IncludedEdge','right'));
edges=[0 998 999];
data.PdaysBin = categorical(discretize(data.pdays,edges,'IncludedEdge','right'));
edges=[-1 0 7];
data.PreviousBin = categorical(discretize(data.previous,edges,'IncludedEdge','right'));
edges=[-3.5 1.1 7];
data.EmpVarRateBin = categorical(discretize(data.emp_var_rate,edges,'IncludedEdge','right'));
cons_price_idx_quantile=quantile(data.cons_price_idx,[0 0.25 0.50 0.75 1]); %use quartiles for 4 bins
data.ConsPriceInxBin = categorical(discretize(data.cons_price_idx,cons_price_idx_quantile,'IncludedEdge','right'));
edges=[-51 -41.8 -25];
data.ConsConfInxBin = categorical(discretize(data.cons_conf_idx,edges,'IncludedEdge','right'));
euribor3m_quantile=quantile(data.euribor3m,[0 0.25 0.50 0.75 1]); %use quartiles for 4 bins
data.euribor3mBin = categorical(discretize(data.euribor3m,euribor3m_quantile,'IncludedEdge','right'));
edges=[4900 5191 5500];
data.nremployedBin = categorical(discretize(data.nr_employed,edges,'IncludedEdge','right'));

data.job = categorical(data.job);
data.marital = categorical(data.marital);
data.education = categorical(data.education);
data.default = categorical(data.default);
data.housing = categorical(data.housing);
data.loan = categorical(data.loan);
data.contact = categorical(data.contact);
data.month = categorical(data.month);
data.day_of_week = categorical(data.day_of_week);
data.poutcome = categorical(data.poutcome);
data.duration = [];

%Recreate y in [0,1] format
y2 = table2array(data(:,20));
y2 = array2table(strcmp(y2,'yes'));
data = [data y2];
data.y = [];

%Create separate tables/vectors for predictor and classification variables
X = data(:,1:end-1);
y = table2array(data(:,end));
end

