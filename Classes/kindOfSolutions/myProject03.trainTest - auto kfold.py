import numpy as np
import matplotlib.pyplot as plt
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

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
    
print('Nearest Neighbour with train test split')
trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(features, labels, test_size=0.4, random_state=42)
model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
y_pred = model.fit(trainFeatures, trainLabels[:,0]).predict(testFeatures)
y_pred = y_pred[:,np.newaxis]
aux = (y_pred!=testLabels)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/testLabels.size
print (misclassificationRate)

print('Nearest Neighbour with 10 fold cross validation')
kf = KFold(Nsamples, n_folds=10)
all_pred, all_y = [], []
for train_index, test_index in kf:
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
    y_pred = model.fit(X_train, y_train[:,0]).predict(X_test)
    y_pred = y_pred[:,np.newaxis]
    all_pred.append(y_pred)
    all_y.append(y_test)
all_pred=np.vstack(all_pred)    
all_y=np.vstack(all_y) 
aux = (all_pred!=all_y)
aux = np.sum(aux.astype(float), 0)
misclassificationRate = aux/all_y.size
print (misclassificationRate)
