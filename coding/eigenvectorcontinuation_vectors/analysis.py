from eigenvec_cont import *
import setupmodels
import matplotlib
import numpy as np
from plot_eigenvectors import plot_eigenvectors
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

models=setupmodels.modelgenerator(500)
model3=models.model3()
c=np.linspace(0,1.5,100)
num_eig=5
print("Start calculating eigenvecotrs")
true_eigenvalues,true_eigenvectors=find_lowest_eigenvectors(model3,c,num_eig,True)
import pickle
dicty={}
dicty["exact eigenvalues"]=true_eigenvalues
Hs=[]
S=np.zeros((len(c),len(c)))
eigenvalues,eigenvectors=find_lowest_eigenvectors(model3,c,1,True)
print("S")

for i in range(len(c)):
    for j in range(i,len(c)):
        S[j,i]=S[i,j]=eigenvectors[i].T@eigenvectors[j]
for cval in c:
    print(cval)
    H=np.zeros((len(c),len(c)))
    for i in range(len(c)):
        for j in range(len(c)):
            H[j,i]=H[i,j]=eigenvectors[i].T@model3(cval)@eigenvectors[j]
    Hs.append(H)
dicty["Hs"]=Hs
dicty["S"]=S
file="analysis_data.bin"
with open(file,"wb") as f:
    pickle.dump(dicty,f)
