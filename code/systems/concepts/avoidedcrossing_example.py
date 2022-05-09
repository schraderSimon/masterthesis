import numpy as np
import matplotlib.pyplot as plt


E1=1
E2=-1
W1=1
W2=-1
M1=M2=np.zeros((2,2))+3
M1[0,0]=E1
M1[1,1]=E2
M1[0,0]=W1
M1[1,1]=W2
theta=0.001
U=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])

M2=U@M2@U.T
eigvals=[]
groundstates=[]
corrfacs=np.linspace(0.5,1.5,501)
overlaps=[]
for i,corrfac in enumerate(corrfacs):
    eigval,eigvec=np.linalg.eigh(M1-corrfac*M2)
    eigvals.append(eigval)
    groundstates.append(eigvec[:,0])
    overlaps.append(abs(groundstates[0].T@groundstates[i])-0.5)
plt.plot(corrfacs,eigvals)
plt.plot(corrfacs,overlaps)
plt.show()
