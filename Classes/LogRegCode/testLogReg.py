import numpy as np
#import matplotlib.pyplot as plt
from logReg import logRegF
    
data = np.genfromtxt('heightWeightData.txt', delimiter=',')
print (np.shape(data))
xx = data[:,1:]
print (np.shape(xx))
yy = data[:,[0]]-1
print (np.shape(yy))
ww = logRegF(xx, yy)
print (np.shape(data))
print (ww)

