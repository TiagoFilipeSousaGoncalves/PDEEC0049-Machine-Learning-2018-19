import numpy as np

#xx = [int(c) for c in '166562663611111122']
xx = [0.7, 0.7, 0.1, 0.2, 0.3, 0.6, 0.2, 0.3, (-0.1), 0.2]
xx = np.array(xx)[:,None]
xx = xx-1 # index starts at 0
print(xx)

AA = np.array([[0.95, 0.05], [0.05, 0.95]]) #transition matrix
pi0 = np.array([0.5, 0.5])[:,None] #initial probs
Px = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]]) # emission probs
T = np.size(xx)

K = 2 # number of states

# forward method - slide 31
alpha=pi0*Px[:,xx[0]]
allAlpha=alpha; # only needed for slide 40
for t in range(1,T):
    alpha=Px[:,xx[t]]*np.dot(AA.T,alpha)
    allAlpha = np.concatenate((allAlpha,alpha),1) # only needed for slide 40

print (allAlpha)
SequenceProbability=np.sum(alpha,0)
print (SequenceProbability)