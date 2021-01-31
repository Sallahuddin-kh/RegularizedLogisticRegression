import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

raw = pd.read_csv('ex2data1.txt', sep=",", header=None, names=["exam1","exam2","pred"])
raw['Q0']=1
features=np.array(raw[["Q0","exam1", "exam2"]].values)
predictions=np.array(raw["pred"].values)
predictions = np.reshape(predictions,newshape=[100,1])
parameters=np.zeros(shape=[3,1])


mean_exam1=np.mean(features[:,1])
mean_exam2=np.mean(features[:,2])
std_dev_exam1=np.std(features[:,1])
std_dev_exam2=np.std(features[:,2])
features[:,1]=(features[:,1]-mean_exam1)/std_dev_exam1
raw["exam1"]=(raw["exam1"]-mean_exam1)/std_dev_exam1
raw["exam2"]=(raw["exam2"]-mean_exam2)/std_dev_exam2
features[:,2]=(features[:,2]-mean_exam2)/std_dev_exam2


iterations = 5000
alphaVal = 0.1
n = len(features)
for i in range(0,iterations):
    g_z = np.dot(features,parameters)
    h_z = 1/(1+np.exp(-g_z))
    temp = h_z - predictions
    cost = np.dot(np.transpose(features), temp)
    parameters -= (alphaVal/n)*(cost)



positives=raw[raw["pred"].isin([1])]
negatives=raw[raw["pred"].isin([0])]
figure, p=plt.subplots(figsize=(10,6))

p.scatter(positives["exam1"], positives["exam2"], color="black", marker="+",label="Admitted")
p.scatter(negatives["exam1"], negatives["exam2"], color="yellow", marker="o",label="Not Admitted")
p.legend()
p.set_xlabel("Exam1")
p.set_ylabel("Exam2")

x_vals = np.array(p.get_xlim())
y_vals = -(x_vals * parameters[1] + parameters[0])/parameters[2]
plt.plot(x_vals, y_vals,  c="blue")

marks1 = (83-mean_exam1)/std_dev_exam1
marks2 = (48-mean_exam2)/std_dev_exam2

probability = 1/(1+np.exp(-(parameters[0]+parameters[1]*marks1+parameters[2]*marks2)))
print(probability) 
if(probability<0.5):
    print("Less likely to pass")
else:
    print("Likely to pass")