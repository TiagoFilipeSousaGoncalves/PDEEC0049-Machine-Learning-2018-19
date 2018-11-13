import numpy as np
import matplotlib.pyplot as plt
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

#data = np.genfromtxt('heightWeightData.txt', delimiter=',')
data = np.genfromtxt('testData.txt', delimiter=';')
Nsamples = np.shape(data)[0]
features = data[:,0:2]
labels = data[:,-1:]
K = np.unique(labels).size

plt.clf() 
lineStyle= ['ob', '*g', '+c', 'xr', '>y']
for cls in range(K):
    idx = (labels == cls+1)
    plt.plot(features[np.nonzero(idx)[0],0], features[np.nonzero(idx)[0],1], lineStyle[cls])
    
print('Support Vector Machine')
trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(features, labels, test_size=0.4, random_state=42)
param_grid = [
  {'C': [1, 10, 100, 1000], 'degree':[1, 2, 3], 'kernel': ['poly'], 'coef0': [1.0]},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
model = GridSearchCV(SVC(), param_grid, cv=5)
model.fit(trainFeatures, trainLabels[:,0])
print("Best parameters set found on development set:")
print()
print(model.best_params_)
y_pred = model.predict(testFeatures)
print(classification_report(testLabels, y_pred))
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=testLabels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/testLabels.size
print (misclassificationRate)