import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def unMap(X, Y):
    deg = 6
    ret = np.ones(1)
    for i in range(1, deg+1):
        for j in range(i+1):
            ret = np.hstack((ret, np.multiply(np.power(X, i-j), np.power(Y, j))))
    return ret
#file Reading stage
raw = pd.read_csv('ex2data2.txt', sep=",", header=None, names=["exam1","exam2","pred"])
raw['Q0']=1
features=np.array(raw[["Q0","exam1", "exam2"]].values)
predictions=np.array(raw["pred"].values)
predictions = np.reshape(predictions,newshape=[118,1])
parameters=np.zeros(shape=[28,1])
positives=raw[raw["pred"].isin([1])]
negatives=raw[raw["pred"].isin([0])]
figure, p=plt.subplots(figsize=(10,6))
x = np.linspace(-1, 1.5, 50)
y = np.linspace(-1, 1.5, 50)
z = np.zeros((len(x), len(y)))
p.scatter(positives["exam1"], positives["exam2"], color="black", marker="+",label="Admitted")
p.scatter(negatives["exam1"], negatives["exam2"], color="yellow", marker="o",label="Not Admitted")
p.legend()
p.set_xlabel("Exam1")
p.set_ylabel("Exam2")
#feature mapping stage
deg = 6
x1 = features[:,1]
x2 = features[:,2]
mapFeature = np.ones(features.shape[0])[:,np.newaxis]    
for n in range(1, deg+1):
        for m in range(n+1):
            mapFeature = np.hstack((mapFeature, np.multiply(np.power(x1, n-m),np.power(x2, m))[:,np.newaxis]))
   
#Regularized linear regression
iterations = 1000
alphaVal = 0.1
n = len(features)
cost_vector = []
iter_vector = []
lambda1 = 1
for i in range(0,iterations):
    g_z = np.dot(mapFeature, parameters)
    h_z=1/(1+np.exp(-g_z))
    pred=h_z-predictions
    hQ = -1/n * np.sum( np.dot(np.transpose(np.log(h_z)), predictions) + np.dot(np.transpose(np.log(1-h_z)), (1-predictions)))
    r = (lambda1/(2*n))*np.sum(np.square(parameters[1:28]))
    hQ = hQ + r 
    gradient = (1/n)*(np.dot(np.transpose(mapFeature), pred))# + r
    parameters -= (alphaVal)*(gradient)
    cost_vector.append(hQ)
    iter_vector.append(i+1)
for n in range(len(x)):
    for m in range(len(y)):
        z[n,m] = np.dot(unMap(x[n], y[m]), parameters)
plt.contour(x,y,z,0)
plt.show()
plt.plot(iter_vector,cost_vector)

