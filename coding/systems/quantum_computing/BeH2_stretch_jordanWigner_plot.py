import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
#from pyscf.mcscf import CASSCF
#from quantum_library import *
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def molecule(x):
    return "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
basis="STO-6G"

file="energy_data/UCC2_BeH2_stretch.bin"
import pickle
with open(file,"rb") as f:
    dicty=pickle.load(f)
E_exact=np.array(dicty["E_FCI"])
dicterino=loadmat("data/BeH2_Jordanwigner_UCCSD2.mat")
UCCSD_energies=dicterino["UCCSD2"][0]
file="energy_data/UCC_BeH2_stretch_JordanWigner_Procrustes.bin"
#file="energy_data/UCC_BeH2_stretch_JordanWigner_genpPocrustes.bin"

with open(file,"rb") as f:
    dictionary=pickle.load(f)
Hs=dictionary["H"]
Ss=dictionary["S"]
x=dictionary["xvals"]
xvals=x
E_FCI=dictionary["E_FCI"]
sample_x=np.array(dictionary["sample_x"])
sample_E=[]
for k,sample_x_val in enumerate(sample_x):
    idx = (np.abs(xvals - sample_x_val)).argmin()
    print(np.abs(xvals - sample_x_val))
    sample_E.append(Hs[idx][k,k])
    print(idx)
    #print(Hs[idx])
    #print(Hs[idx][k,k])
sample_E=np.array(sample_E)
#sample_E=np.array(dictionary["sample_E"])
CCEVC_energies=[]
#sample_points=[[1,2,3,4,5,6,7,8],[0,14]]
sample_points=[[0,10,11,12],[0,10,11,12,14]]
for i in range(2):
    vals=sample_points[i]
    E=[]
    for k in range(len(xvals)):
        H=Hs[k][np.ix_(vals,vals)].copy()
        S=Ss[k][np.ix_(vals,vals)].copy()
        e,c=canonical_orthonormalization(H,S,threshold=1e-16) #lazy way to solve generalized eigenvalue problem
        E.append(e)
    for eigval in np.linalg.eigh(S)[0]:
        print(eigval)
    CCEVC_energies.append(E)
fig,axes=plt.subplots(1,2,sharey=True,sharex=True,figsize=(12,3))
axes[0].set_ylabel("Energy (Hartree)")
axes[0].set_xlabel("x (Bohr)")
axes[1].set_xlabel("x (Bohr)")

for i in range(2):
    axes[i].plot(x,E_FCI,label="FCI",color="tab:purple",alpha=0.7)
    axes[i].plot(x,UCCSD_energies,label="1-UCCSD",color="tab:blue",alpha=1)
    if i==0:
        pass
    axes[i].plot(x,CCEVC_energies[i],"--",label="EVC",color="tab:orange",alpha=0.9)
    axes[i].plot(sample_x[sample_points[i]],sample_E[sample_points[i]],"*",label="Smp. pts.",color="black",markersize=9)
    axes[i].grid()
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.905,0.58),loc="center",handletextpad=0.3,labelspacing = 0.1)

plt.tight_layout()
fig.subplots_adjust(right=0.81)
plt.savefig("resultsandplots/BeH2_stretch_QC_fix.pdf")

plt.show()
