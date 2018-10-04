import numpy as np
import matplotlib.pyplot as plt

#USING OUR OWN IMPLEMENTATION OF LINEAR REGRESSION
def polyRegression(data1D, yy, testData, degree):
    xdata = [data1D**dd for dd in range (degree+1)]
    xdata = np.concatenate(xdata, axis=1)
    
    ww = np.linalg.inv(np.dot(xdata.transpose(),xdata))
    ww = np.dot(ww, xdata.transpose())
    ww = np.dot(ww, yy)    
    
    xdata = [testData**dd for dd in range (degree+1)]
    xdata = np.concatenate(xdata, axis=1)
    pred = np.dot(xdata, ww)    
    return pred, ww
    
        
    
data = np.genfromtxt('data_bishop.txt', delimiter=' ')
print (np.shape(data))
print (type(data))
print (data)
testData = np.linspace(0, 1, 500).reshape(500,1)
pred, model  = polyRegression(data[:,[0]], data[:,[-1]], testData, 3)
print (np.shape(pred))
print (type(pred))
plt.plot(testData, pred);
plt.plot(data[:,[0]], data[:,[-1]], 'o')
print (model)


#USING SKLEARN
print ("USING sklearn")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = np.genfromtxt('data_bishop.txt', delimiter=' ')
print (np.shape(data))
print (type(data))
print (data)
testData = np.linspace(0, 1, 500).reshape(500,1)
model = make_pipeline(PolynomialFeatures(9), LinearRegression())
model = model.fit(data[:,[0]], data[:,-1])
pred = model.predict(testData)
print (np.shape(pred))
print (type(pred))

plt.plot(testData, pred);
plt.plot(data[:,[0]], data[:,[-1]], 'o')
print (model)
