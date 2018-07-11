function [confMat,Accuracy,Error,Sensitivity,Specificity,Precision] = confusion(y,prediction)
%Author - Toby Staines

%Produce a confusion Matrix using the supplied true classifications(y) and
%predictions (prediction)
confMat = [sum(y == 0 & prediction == 0) sum(y == 0 & prediction == 1);
           sum(y == 1 & prediction == 0) sum(y == 1 & prediction == 1)];

%Calculate the accuracy, error, sensitivity and specificity rates of the model       
Accuracy = (confMat(1,1) + confMat(2,2))/length(y);
Error = (confMat(1,2) + confMat(2,1))/length(y);
Sensitivity = confMat(2,2)/(confMat(2,2)+confMat(2,1));
Specificity = confMat(1,1)/(confMat(1,1)+confMat(1,2));
Precision = confMat(2,2)/(confMat(2,2)+confMat(1,2));
end

