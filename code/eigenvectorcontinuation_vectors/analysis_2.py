from eigenvec_cont import *
import setupmodels
import matplotlib
import numpy as np
import scipy.linalg as linalg
from plot_eigenvectors import plot_eigenvectors
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

import pickle
file="analysis_data.bin"
with open(file,"rb") as f:
    data=pickle.load(f)
S=data["S"]
Hs=data["Hs"]
vals=[0,20,40,60,80]
eigvals=[]
c=np.linspace(0,1.5,100)
for i,cval in enumerate(c):
    H=Hs[i][np.ix_(vals,vals)]
    print(Hs[i][i,i])
    Ss=S[np.ix_(vals,vals)]
    exponent=8
    eigval=linalg.eigh(a=H,b=Ss)[0][0]
    eigvals.append(eigval)
plt.plot(eigvals)
plt.show()
