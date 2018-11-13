import numpy as np
import matplotlib.pyplot as plt
#from sklearn.qda import QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#data = np.genfromtxt('heightWeightData.txt', delimiter=',')
data = np.genfromtxt('testData.txt', delimiter=';')
Nsamples = np.shape(data)[0]
features = data[:,0:2]
labels = data[:,-1:]
trainFeatures=features[:round(0.6*Nsamples),:] 
trainLabels=labels[:round(0.6*Nsamples),:] 
testFeatures=features[round(0.6*Nsamples):,:] 
testLabels=labels[round(0.6*Nsamples):,:] 
print(np.shape(trainFeatures), np.shape(testFeatures), np.shape(trainLabels), np.shape(testLabels))

K = np.unique(labels).size

plt.clf() 
lineStyle= ['ob', '*g', '+c', 'xr', '>y']
for cls in range(K):
    idx = (labels == cls+1)
    plt.plot(features[np.nonzero(idx)[0],0], features[np.nonzero(idx)[0],1], lineStyle[cls])
    
print('Discriminant analysis')
model = QuadraticDiscriminantAnalysis()
y_pred = model.fit(trainFeatures, trainLabels[:,0]).predict(testFeatures)
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=testLabels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/testLabels.size
print (misclassificationRate)

print('Logistic Regression')
model = LogisticRegression(multi_class = 'multinomial', solver='newton-cg', C=100)
#create extended features
xtrainFeatures = np.concatenate((trainFeatures, trainFeatures[:,0:1]*trainFeatures[:,1:2]), 1)
xtestFeatures = np.concatenate((testFeatures, testFeatures[:,0:1]*testFeatures[:,1:2]), 1)
y_pred = model.fit(xtrainFeatures, trainLabels[:,0]).predict(xtestFeatures)
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=testLabels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/testLabels.size
print (misclassificationRate)

print('Nearest Neighbour')
model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
y_pred = model.fit(trainFeatures, trainLabels[:,0]).predict(testFeatures)
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=testLabels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/testLabels.size
print (misclassificationRate)

print('Support Vector Machine')
model = SVC(kernel = 'poly', degree=2, coef0=1.0, C=100)
y_pred = model.fit(trainFeatures, trainLabels[:,0]).predict(testFeatures)
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=testLabels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/testLabels.size
print (misclassificationRate)

