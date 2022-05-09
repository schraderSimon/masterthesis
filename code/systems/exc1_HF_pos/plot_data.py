import sys
sys.path.append("../../eigenvectorcontinuation/")
import matplotlib
from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
sample_geometry=[[np.linspace(1.5,2.0,3),np.linspace(1.5,5.0,3)],[np.linspace(1.5,2.0,3),np.linspace(1.5,5,3)]]
types=["procrustes","transform"]
name=["Gen. Procrustes","Symm. Orth."]
basis="cc-pVDZ"
HF_STO3G=[-98.2949,-98.4312,-98.5101,-98.5523,-98.5702,-98.572 ,-98.563 ,-98.5469,-98.5264,-98.5031,-98.4783,-98.4528,-98.427 ,-98.4013,-98.3763,-98.352 ,-98.3289,-98.307 ,-98.2864,-98.2673,-98.2497,-98.2334
,-98.2185,-98.2049,-98.1925,-98.1812,-98.1709,-98.1616,-98.1532,-97.8139]
HF_631G=[-99.782 ,-99.8907,-99.9484,-99.9753,-99.9833,-99.9799,-99.9694,-99.9547,-99.9376,-99.9194,-99.9007,-99.882 ,-99.8636,-99.8457,-99.8285,-99.812 ,-99.7962,-99.7811,-99.7669,-99.7534,-99.7407,-99.7286
,-99.7173,-99.7067,-99.6967,-99.6873,-99.6784,-99.6701,-99.6623,-99.655 ]
HF_ccpvdz=[-99.8288 ,-99.9157 ,-99.9694,-100.0002,-100.0153,-100.0197,-100.0168,-100.0089 ,-99.9977 ,-99.9844 ,-99.9698 ,-99.9545 ,-99.9389 ,-99.9232 ,-99.9078 ,-99.8926 ,-99.8778 ,-99.8636 ,-99.8498
 ,-99.8365 ,-99.8238 ,-99.8116 ,-99.7999 ,-99.7887 ,-99.7781 ,-99.7679 ,-99.7581 ,-99.7488 ,-99.74   ,-99.7315 ,-99.7235 ,-99.7158 ,-99.7084 ,-99.7014 ,-99.6948 ,-99.6885 ,-99.6824 ,-99.6767
 ,-99.6712]
CC_ccpvdz=[-100.0256,-100.1148,-100.1708,-100.2038,-100.2212,-100.2277,-100.227 ,-100.2213,-100.2123,-100.2013,-100.189 ,-100.176 ,-100.1629,-100.1499,-100.1373,-100.1252,-100.1136,-100.1028,-100.0927,-100.0833,-100.0746,-100.0667,-100.0595,-100.053 ,-100.0471,-100.0418,-100.0371,-100.033 ,-100.0293,-100.0261,-100.0233,-100.0208,-100.0187,-100.0168,-100.0152,-100.0138,-100.0125,-100.0114,-100.0105]
threshold=1e-16
sample_geometry=np.linspace(1.5,2.0,3)
energies_sample=[-100.00017152974473, -100.01898111740954, -99.99772118536649]
titles=["Gen. Procrustes","Symm. Orth."]
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("x (Bohr)")
axes[1][1].set_xlabel("x (Bohr)")
axes[1][1].set_ylim([-100.25,-99.6])
axes[0][0].set_xticks(np.linspace(2,5,4))

for i in range(2):
    for j in range(2):
        file="HF_%s_%d%d.bin"%(basis,i,j)
        import pickle
        with open(file,"rb") as f:
            data=pickle.load(f)
        Hs=data["Hs"]
        Ss=data["Ss"]
        xc_array=data["xc_array"]
        energies_1=[]
        energies_2=[]
        energies_3=[]
        lenny=len(Hs[0])
        print(lenny)
        for k in range(len(xc_array)):
            print(k)
            H=Hs[k]
            S=Ss[k]
            eigval,eigvec=generalized_eigenvector(H[:lenny//3,:lenny//3],S[:lenny//3,:lenny//3],threshold)
            energies_1.append(eigval)
            eigval,eigvec=generalized_eigenvector(H[:2*lenny//3,:2*lenny//3],S[:2*lenny//3,:2*lenny//3],threshold)
            energies_2.append(eigval)
            eigval,eigvec=generalized_eigenvector(H,S,threshold)
            energies_3.append(eigval)
        print(energies_1)
        axes[i][j].set_title(titles[i])
        axes[i][j].plot(xc_array,HF_ccpvdz,label="RHF",color="tab:cyan")
        axes[i][j].plot(xc_array,CC_ccpvdz,label="CCSD",color="tab:purple")
        axes[i][j].plot(xc_array,energies_1,"--",label="EVC (1 pt.)",color="tab:orange")
        axes[i][j].plot(xc_array,energies_2,"--",label="EVC (2 pt.)",color="tab:red")
        axes[i][j].plot(xc_array,energies_3,"--",label="EVC (3 pt.)",color="tab:green")

        axes[i][j].grid()
        axes[i][j].plot(sample_geometry,energies_sample,"*",color="black",label="Sample pts.",markersize=9)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.0,0.450),loc="lower right",handletextpad=0.3,labelspacing = 0.1)
fig.tight_layout()
fig.subplots_adjust(right=0.85)
plt.savefig("HF_EVC_singles.pdf")

plt.show()
