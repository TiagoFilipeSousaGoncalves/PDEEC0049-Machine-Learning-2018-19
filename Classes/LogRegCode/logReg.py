import numpy as np

def logRegF(XX, yy):
    NN, DD = np.shape(XX)
    xdata = np.concatenate((np.ones((NN,1)), XX), axis=1)
    ww = np.zeros((DD+1,1))
    for i in range(1000)    :
        mu = 1/(1+np.exp(-np.dot(xdata,ww)))    
        S = np.diag(np.multiply(mu,(1-mu))[:,0])
        aux = np.linalg.inv(np.dot(np.dot(xdata.T,S),xdata))
        aux = np.dot(np.dot(aux,xdata.T), yy-mu)
        ww = ww + aux
    return ww 

