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
#plot_eigenvectors(model3,c,symmetric=True,num=6)
colors=["red","blue","green","brown","orange","orange"]
maxnum=2
num_eig=5
true_eigenvalues,true_eigenvectors=find_lowest_eigenvectors(model3,c,num_eig,True)


print(true_eigenvalues)
eigvals=np.zeros(len(c))
counter=0
sample_vals=np.linspace(0,1.5,100)
fig,ax=plt.subplots(1,maxnum,sharex=True,sharey=True,figsize=(10,4))
true_sample_eigvals,trash=find_lowest_eigenvectors(model3,sample_vals,1,True)
ax[0].plot(c,true_eigenvalues[::num_eig],label=r"True $\lambda_0(c)$",color=colors[0])
for i in range(0,len(sample_vals),2):
    x=sample_vals[:i+1]
    eigenvalues,eigenvectors=find_lowest_eigenvectors(model3,x,1,True)
    for k,xval in enumerate(c):
        eigvals[k],soppel=eigenvec_cont(xval,model3,eigenvectors) #Find eigenvalues
    ax[0].plot(c,eigvals,"--",label="EC, %d sampl. point(s)"%(i+1),color=colors[i+1])
for i in range(num_eig):
    plt.plot(c,true_eigenvalues[i::num_eig],label=r"True $\lambda_%d(c)$"%i,color=colors[i])
ax[0].set_ylabel(r"$\lambda(c)$")
ax[1].set_xlabel(r"c")
ax[0].set_xlabel(r"c")
ax[1].set_ylim(0,150)
ax[0].plot(sample_vals,true_sample_eigvals,"o",color="black",label="sample points")
ax[0].legend(loc="lower left")
ax[1].legend(loc="lower left")
plt.tight_layout()
plt.savefig("present1.pdf")
plt.show()
