from eigenvec_cont import *
import setupmodels
import numpy as np
from plot_eigenvectors import plot_eigenvectors
models=setupmodels.modelgenerator(500)
model3=models.model3()
c=np.linspace(0,1.5,50)
#plot_eigenvectors(model3,c,symmetric=True,num=6)

sample_vals=np.array([0,0.9,1.1])
maxnum=3
true_eigenvalues,true_eigenvectors=find_lowest_eigenvectors(model3,c,1,True)
eigvals=np.zeros(len(c))
fig,ax=plt.subplots(1,maxnum,sharex=True,sharey=True)
for i in range(0,len(sample_vals)):
    x=sample_vals[:(i+1)]
    for j in range(1,maxnum+1):
        eigenvalues,eigenvectors=find_lowest_eigenvectors(model3,x,j,True)
        for k,xval in enumerate(c):
            eigvals[k],soppel=eigenvec_cont(xval,model3,eigenvectors)
        ax[j-1].plot(c,eigvals,label="Continuation with i=%d"%i)
for j in range(maxnum):
    ax[j].plot(c,true_eigenvalues,label="True")
    ax[j].set_title("Eigenvalues up to %d excitation")
    ax[j].legend()

plt.show()
