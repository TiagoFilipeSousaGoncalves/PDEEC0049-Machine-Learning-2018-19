function Prediction = knnClassifier (testData, trainFeatures, trainLabels, k)
N = size(trainFeatures,1); % number of training instances
M = size(testData,1); % number of test instances
Prediction = zeros(M,1); % initialize prediction vector
for i = 1:M
    Distances = sqrt(sum((repmat(testData(i,:),N,1)-trainFeatures).^2,2));               
    sortedDistances = sortrows([Distances trainLabels],1);
    Prediction(i) = mode(sortedDistances(1:k,2));
end