from eigenvec_cont import *
import setupmodels
import numpy as np
from plot_eigenvectors import plot_eigenvectors
from celluloid import Camera
models=setupmodels.modelgenerator(500)
model3=models.model3()
c=np.linspace(0,1.5,50)
#plot_eigenvectors(model3,c,symmetric=True,num=6)

maxnum=3
true_eigenvalues,true_eigenvectors=find_lowest_eigenvectors(model3,c,1,True)
eigvals=np.zeros(len(c))

counter=0
for samp in np.linspace(0.05,0.75,50):
    fig,ax=plt.subplots(1,maxnum,sharex=True,sharey=True,figsize=(10,10))
    sample_vals=np.array([0,samp,2*samp])
    true_sample_eigvals,trash=find_lowest_eigenvectors(model3,sample_vals,1,True)
    fig.suptitle(r"EC with points $\{%.2f, %.2f, %.2f\}$"%(sample_vals[0],sample_vals[1],sample_vals[2]))

    for i in range(0,len(sample_vals)):
        x=sample_vals[:(i+1)]
        for j in range(1,maxnum+1):
            eigenvalues,eigenvectors=find_lowest_eigenvectors(model3,x,j,True)
            for k,xval in enumerate(c):
                eigvals[k],soppel=eigenvec_cont(xval,model3,eigenvectors)
            ax[j-1].plot(c,eigvals,label="EC with i=%d sample points"%i)
            ax[j-1].plot(sample_vals,true_sample_eigvals,"o",color="black")
    for j in range(maxnum):
        ax[j].set_xlabel("c")
        ax[j].plot(c,true_eigenvalues,label="True")
        ax[j].set_title("%d ex. vector per c"%j)
        ax[j].legend()
        ax[j].set_ylabel(r"$\lambda_0(c)$")
    plt.savefig("eigvec_cont_%03d.png"%counter)
    counter+=1
    plt.close(fig)
