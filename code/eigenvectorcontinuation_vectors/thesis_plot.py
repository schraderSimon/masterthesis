from eigenvec_cont import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
import setupmodels
from plot_eigenvectors import plot_eigenvectors


models=setupmodels.modelgenerator(500)
model3=models.model3()
start=0.3
c=np.linspace(start,1.5,111)
#plot_eigenvectors(model3,c,symmetric=True,num=6)
colors=["tab:blue","tab:green","tab:purple","tab:orange","tab:red"]
maxnum=2
num_eig=2
true_eigenvalues,true_eigenvectors=find_lowest_eigenvectors(model3,c,num_eig,True)
true_eigenvalues_=[]
true_eigenvectors_=[]
for i in range(num_eig):
    print(i)
    true_eigenvalues_.append(true_eigenvalues[i::num_eig])
    true_eigenvectors_.append(true_eigenvectors[i::num_eig])
true_eigenvalues_=np.array(true_eigenvalues_)
true_eigenvectors_=np.array(true_eigenvectors_)
#plt.plot(c,true_eigenvalues_[1]-true_eigenvalues_[0])
#plt.show()
fig,ax=plt.subplots(2,2,sharex=True,sharey=False,figsize=(10,12))
for i in range(2):
    for j in range(2):
        style="--"
        l=1
        ax[i][j].axvline(0.834,linestyle=style,color="grey",linewidth=l,label="Av. cross.")
        ax[i][j].axvline(1.00,linestyle=style,color="grey",linewidth=l)
        ax[i][j].axvline(1.25,linestyle=style,color="grey",linewidth=l)


counter=0
num_samp=5
sample_vals_spread=np.linspace(start,1.5,num_samp)
sample_eigvals_spread,sample_eigvecs_spread=find_lowest_eigenvectors(model3,sample_vals_spread,1,True)

sample_vals=np.linspace(start,start+0.3,num_samp)
sample_eigvals_tight,sample_eigvecs_tight=find_lowest_eigenvectors(model3,sample_vals,1,True)
nums=[1,3,5]
ax[0][0].set_ylabel(r"$\lambda(\alpha)$")
ax[1][0].set_ylabel(r"$|\langle\psi^{EVC}(\alpha)|\psi_0(\alpha)\rangle|^2$")
ax[1][0].set_xlabel(r"$\alpha$")
ax[1][1].set_xlabel(r"$\alpha$")

ax[0][0].plot(c,true_eigenvalues_[0],label=r"True $\lambda_0(\alpha)$",color=colors[0])
ax[0][0].plot(c,true_eigenvalues_[1],label=r"True $\lambda_1(\alpha)$",color=colors[1])
ax[0][1].plot(c,true_eigenvalues_[0],label=r"True $\lambda_0(\alpha)$",color=colors[0])
ax[0][1].plot(c,true_eigenvalues_[1],label=r"True $\lambda_1(\alpha)$",color=colors[1])
alpha=0.8
eigvals_evc=np.zeros(len(c))
overlaps=np.zeros(len(c))
threshold=1e-15
for i,num in enumerate(nums):
    for k,xval in enumerate(c):
        eigvals_evc[k],eigvec_evc=eigenvec_cont(xval,model3,sample_eigvecs_tight[:num],threshold=threshold) #Find eigenvalues
        evc_eigvec=np.einsum("i,ia->a",eigvec_evc,sample_eigvecs_tight[:num])
        overlaps[k]=np.abs((evc_eigvec.T@true_eigenvectors_[0,k,:]))**2
    ax[0][0].plot(c,eigvals_evc,"--",label="EVC, %d s. pt."%(num),color=colors[i+2],alpha=alpha)
    ax[1][0].plot(c,overlaps,"--",label=r"overlap to $\psi_0$, %d smp. pt."%(num),color=colors[i+2],alpha=1)
for i,num in enumerate(nums):
    for k,xval in enumerate(c):
        eigvals_evc[k],eigvec_evc=eigenvec_cont(xval,model3,sample_eigvecs_spread[:num],threshold=threshold) #Find eigenvalues
        evc_eigvec=np.einsum("i,ia->a",eigvec_evc,sample_eigvecs_spread[:num])
        overlaps[k]=np.abs((evc_eigvec.T@true_eigenvectors_[0,k,:]))**2
    ax[0][1].plot(c,eigvals_evc,"--",label="EVC, %d s. pt."%(num),color=colors[i+2],alpha=alpha)
    ax[1][1].plot(c,overlaps,"--",label=r"overlap to $\psi_0$, %d smp. pt."%(num),color=colors[i+2],alpha=1)
m=7
ax[0][0].plot(sample_vals,sample_eigvals_tight,"*",label=r"sample pts.",color="black",markersize=m)
ax[1][0].plot(sample_vals,np.ones(len(sample_vals)),"*",label=r"sample pts.",color="black",markersize=m)
handles, labels = ax[0][0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.4,0.54),loc="lower right",handletextpad=0.3,labelspacing = 0.1)
ax[0][1].plot(sample_vals_spread,sample_eigvals_spread,"*",label=r"sample pts.",color="black",markersize=m)
ax[1][1].plot(sample_vals_spread,np.ones(len(sample_vals_spread)),"*",label=r"sample pts.",color="black",markersize=m)
ax[1][1].set_yticks([])
ax[0][1].set_yticks([])

plt.tight_layout()
plt.savefig("present1.pdf")
plt.show()
