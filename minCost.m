function [minCostThreshold, minCidx] = minCost(alpha,beta,y,rocX,rocY,rocT)
%Author - Toby staines

%Calculates the classification threshold probability with the lowest cost,
%given roc coordinates, true classifications and cost weight parameters

% Apha and beta are parameters which give relative weight to the cost of
% false positives and false negatives respectively
% p is the prior probability of a positive result
% C = (1-p) alpha x + p beta (1-y)

p = sum(y)/size(y,1); % Prior - Proportion of positive cases in data set
C = ((1-p)*alpha*rocX)+(p*beta*(1-rocY)); % Vector of cost for each point on the ROC curve
[~, minCidx] = min(C);% The index of the point with the minimum cost
minCostThreshold = rocT(minCidx);% The classification threshold probability with the minimum associated cost
end

