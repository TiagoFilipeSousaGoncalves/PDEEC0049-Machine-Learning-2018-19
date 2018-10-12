# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:37:57 2018

@author: Tiago Filipe Sousa Gonçalves | 5º Ano MIB | UP201607753
"""

#Import libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

"""1. Write a Python function to compute the predictions according to the mean Euclidean distance to the sample points of each class.
The function should have the following interface function [prediction] = meanPrediction(dataClass1, dataClass2, dataUnknownClass) where dataClass1 is an array N1xd; dataClass2 is an array N2xd; dataUnknownClass is an array Ntxd; and prediction is an array Ntx1. d is the dimension of the features.

a)Determine the training error on your samples using only the x1 feature value. Make use of the function meanPrediction you wrote.

b) Repeat but now use two feature values, x1 and x2.

c) Repeat but use all three feature values.

d) Discuss your results. Is it ever possible for a finite set of data that the training error be larger for more data dimensions?"""

#Implement Euclidean Distance Function First
def euclidean_distance(dataClass, dataUnknownClassElement):
    # Computes and returns the Euclidean distance between elements from UnknownDataClass
    #And DataClass
    #Assign temporary distance
    d=np.zeros((dataClass.shape[0],1))
    for index in range(dataClass.shape[0]):
        #In case dataClass has just one column (one feature):
        if dataClass.shape[1]==1:
             d[index]=np.sqrt(np.power((dataUnknownClassElement-dataClass[index]),2))
        
        #For more features     
        else:
             d[index]=np.sqrt(sum(np.power((dataUnknownClassElement-dataClass[index]),2)))
    distance = np.mean(d)        
    
    return distance 

#Implement meanPrediction Function
def meanPrediction(dataClass1, dataClass2, dataUnknownClass):
    #Initialize Distance Arrays
    dc1=np.zeros((dataUnknownClass.shape[0],1))
    dc2=np.zeros((dataUnknownClass.shape[0],1))
    predictions = np.zeros((dataUnknownClass.shape[0], 1))
    
    #Iterate from every dataUnknownClass item
    for index in range(dataUnknownClass.shape[0]):
        #Calculate distances from points to every class and assign in variables dc1 and dc2
        dc1[index] = euclidean_distance(dataClass1, dataUnknownClass[index])
        dc2[index] = euclidean_distance(dataClass2, dataUnknownClass[index])
        
        #Evaluate Classes based on the means
        if np.mean(dc1[index]) < np.mean(dc2[index]):
            predictions[index] = 1
        elif np.mean(dc1[index]) > np.mean(dc2[index]):
            predictions[index] = 2
        else:
            #Randomly assign prediction if the means are equal
            predictions[index] = random.randint(1,2)
            
    return predictions

#Let's creat a function to determine the accuracy of our algorithm
def accuracy(y_test, y_pred):
    #Number of corrected predictions
    corr = 0
    for index in range(y_test.shape[0]):
        if y_test[index] == y_pred[index]:
            corr +=1 
    #corr = int(np.sum(y_test==y_pred))
    n_samples = int((y_test.shape[0]))
    acc = corr/n_samples
    return acc*100


###############################################################################
#Data
dataClass1 = np.zeros((10, 4))
#Assign Values dataClass1
dataClass1[0, 0] = -5.01
dataClass1[0, 1] = -8.12
dataClass1[0, 2] = -3.68

dataClass1[1, 0] = -5.43
dataClass1[1, 1] = -3.48
dataClass1[2, 0] = 1.08
dataClass1[2, 1] = -5.52
dataClass1[2, 2] = -1.66

dataClass1[3, 0] = 0.86
dataClass1[3, 1] = -3.78
dataClass1[3, 2] = -4.11

dataClass1[4, 0] = -2.67
dataClass1[4, 1] = -0.63
dataClass1[4, 2] = 7.39

dataClass1[5, 0] = 4.94
dataClass1[5, 1] = 3.29
dataClass1[5, 2] = 2.08

dataClass1[6, 0] = -2.51
dataClass1[6, 1] = 2.09
dataClass1[6, 2] = -2.59

dataClass1[7, 0] = -2.25
dataClass1[7, 1] = -2.13
dataClass1[7, 2] = -6.94

dataClass1[8, 0] = 5.56
dataClass1[8, 1] = 2.86
dataClass1[8, 2] = -2.26

dataClass1[9, 0] = 1.03
dataClass1[9, 1] = -3.33
dataClass1[9, 2] = 4.33



dataClass2 = np.zeros((10, 4))
#Assign Values to dataClass2
dataClass2[0, 0] = -0.91
dataClass2[0, 1] = -0.18
dataClass2[0, 2] = -0.05

dataClass2[1, 0] = 1.30
dataClass2[1, 1] = -2.06
dataClass2[1, 2] = -3.53

dataClass2[2, 0] = -7.75
dataClass2[2, 1] = -4.54
dataClass2[2, 2] = -0.95

dataClass2[3, 0] = -5.47
dataClass2[3, 1] = 0.50
dataClass2[3, 2] = 3.92

dataClass2[4, 0] = 6.14
dataClass2[4, 1] = 5.72
dataClass2[4, 2] = -4.85

dataClass2[5, 0] = 3.60
dataClass2[5, 1] = 1.26
dataClass2[5, 2] = 4.36

dataClass2[6, 0] = 5.37
dataClass2[6, 1] = -4.63
dataClass2[6, 2] = -3.65

dataClass2[7, 0] = 7.18
dataClass2[7, 1] = 1.46
dataClass2[7, 2] = -6.66

dataClass2[8, 0] = -7.39
dataClass2[8, 1] = 1.17
dataClass2[8, 2] = 6.30

dataClass2[9, 0] = -7.50
dataClass2[9, 1] = -6.32
dataClass2[9, 2] = -0.31

#Labels for Each Classes
#dataClass1 Label is 1
dataClass1[:, 3] = 1
#dataClass2 Label is 2
dataClass2[:, 3] = 2

#dataUnknownClass is the concatenations of both classes:
dataUnknown = np.concatenate((dataClass1, dataClass2), axis=0)

#Print both datasets
print("dataClass1 is : \n", dataClass1, '\n')
print("dataClass2 is : \n", dataClass2, '\n')

#Generate dataUnknownClass
#dataUnknown = np.random.rand(10,3)
print("dataUnknown is: \n", dataUnknown, '\n')

#print(dataClass1.shape[0], dataClass2.shape, dataUnknown.shape)
#print(np.concatenate((dataClass1, dataClass2), axis=0))

#Let's Predict the Unknown Class and check the training error
y_test = np.array(dataUnknown[:, [-1]]      )

#Using Just 1 Feature
y_pred = meanPrediction(dataClass1[:, [0]], dataClass2[:, [0]], dataUnknown[:, [0]])
acc_1f = accuracy(y_test=y_test, y_pred=y_pred)
print("The accuracy with one feature is: ", acc_1f, " % " "with the following predictions: ", y_pred)

#Using Just 2 Feature
#print(dataUnknown)
y_pred = meanPrediction(dataClass1[:, 0:2], dataClass2[:, 0:2], dataUnknown[:, 0:2])
acc_2f = accuracy(y_test=y_test, y_pred=y_pred)
print("The accuracy with two features is: ", acc_2f, " % " "with the following predictions: ", y_pred)

#Using Just 3 Features
#Compute function
y_pred = meanPrediction(dataClass1[:, 0:3], dataClass2[:, 0:3], dataUnknown[:,0:3])
acc_3f = accuracy(y_test, y_pred)
print("The accuracy with three features is ", acc_3f, "% " "with the following predictions: ", y_pred)

print("\nMy thoughts on question d) are:")
print("Answer: Actually, that is one of the problems with this algorithm, i.e., the accuracy of k-NN can be severely degraded with high-dimension data because there is little difference between the nearest and furthest neighbour.\nAlso, one of the suggestions in order to improve the algorithm is to implement dimensionality reduction techniques like PCA, prior to appplying k-NN and help make the distance metric more meaningful.")

###############################################################################

""" 2. Peter is a very predictable man. When he uses his tablet, all he does is watch movies. He always watches until his battery dies. He is also a very meticulous man. He has kept logs of every time he has charged his tablet, which includes how long he charged his tablet for and how long he was able to watch movies for afterwards. Now, Peter wants to use this log to predict how long he will be able to watch movies for when he starts so that he can plan his activities after watching his movies accordingly.
You will be able to access Peter’s tablet charging log by reading from the file “TabletTrainingdata.txt”. The training data file consists of 100 lines, each with 2 comma-separated numbers. The first number denotes the amount of time the tablet was charged and the second denotes the amount of time the battery lasted.
Read an input (test case) from the console (stdin) representing the amount of time the tablet was charged and output to the console the amount of time you predict his battery will last.

    #example to read test case
    timeCharged = float(input().strip())
    
    #example to output
    print(prediction)"""

#Regression Model
#USING OUR OWN IMPLEMENTATION OF LINEAR REGRESSION: Method learned in class
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
    
        
    
data = np.genfromtxt('TabletTrainingdata.txt', delimiter=',')
#print (np.shape(data))
#print (type(data))
#print (data)
print("Insert charged time, in hours:")
timeCharged = float(input().strip())
#I used as input: 5.0

#We have "virtual feature = 1"
testData = np.array([[1], [timeCharged]])

prediction, model  = polyRegression(data[:,[0]], data[:,[-1]], testData, 4)
#print (np.shape(pred))
#print (type(pred))
#plt.plot(testData, pred);
#plt.plot(data[:,[0]], data[:,[-1]], 'o')
#print (model)

print("For ", float(timeCharged), "  hours of charging, the battery will last about ", float(prediction[1]), " hours.")

#Another way of solving this
#Depending on the degree of polynome we can make a simple for cycle to use the weights and to multiply by our input:
#SUM(weight(i)+input**(i))
#print(model)
pred = 0
print("Insert charged time, in hours:")
timeCharged = float(input().strip())

#We iterate through all the weights of the model
for i in range(int(model.shape[0])):
    pred += model[i]*(timeCharged**i)

#Print prediction
print("For ", timeCharged, " hours of charging, the battery will last about ", float(pred), " hours.")

#USING SKLEARN
print ("USING sklearn")

data = np.genfromtxt('TabletTrainingdata.txt', delimiter=',')
#print (np.shape(data))
#print (type(data))
#print (data)

print("Insert charged time, in hours:")
timeCharged = float(input().strip())
#I used as input: 5.0

#We have "virtual feature = 1"
testData = np.array([[1], [timeCharged]])

model = make_pipeline(PolynomialFeatures(4), LinearRegression())
model = model.fit(data[:,[0]], data[:,-1])

prediction = model.predict(testData)
#print (np.shape(pred))
#print (type(pred))
#plt.plot(testData, pred);
#plt.plot(data[:,[0]], data[:,[-1]], 'o')
#print (model)

print("For ", float(timeCharged), "  hours of charging, the battery will last about ", float(prediction[1]), " hours.")