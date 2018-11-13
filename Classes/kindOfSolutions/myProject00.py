import numpy as np
import matplotlib.pyplot as plt
#from sklearn.qda import QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#data = np.genfromtxt('heightWeightData.txt', delimiter=',')
data = np.genfromtxt('testData.txt', delimiter=';')
features = data[:,0:2]
labels = data[:,-1:]

K = np.unique(labels).size

plt.clf() 
lineStyle= ['ob', '*g', '+c', 'xr', '>y']
for cls in range(K):
    idx = (labels == cls+1)
    plt.plot(features[np.nonzero(idx)[0],0], features[np.nonzero(idx)[0],1], lineStyle[cls])
    
print('Discriminant analysis')
model = QuadraticDiscriminantAnalysis()
y_pred = model.fit(features, labels[:,0]).predict(features)
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=labels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/labels.size
print (misclassificationRate)

print('Logistic Regression')
model = LogisticRegression(multi_class = 'multinomial', solver='newton-cg', C=100)
#create extended features
xfeatures = np.concatenate((features, features[:,0:1]*features[:,1:2]), 1)
y_pred = model.fit(xfeatures, labels[:,0]).predict(xfeatures)
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=labels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/labels.size
print (misclassificationRate)

print('Nearest Neighbour')
model = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
model.fit(features, labels[:,0])
y_pred = model.predict(features)
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=labels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/labels.size
print (misclassificationRate)

print('Support Vector Machine')
model = SVC(kernel = 'poly', degree=2, coef0=1.0, C=100)
y_pred = model.fit(features, labels[:,0]).predict(features)
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=labels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/labels.size
print (misclassificationRate)

