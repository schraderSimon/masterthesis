import sys
sys.path.append("../../eigenvectorcontinuation/")
import matplotlib
from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

basis="cc-pVTZ"
sample_geometry=[[np.linspace(1.5,2.0,6),np.linspace(4.5,5,6)],[np.linspace(1.5,2.0,6),np.linspace(4.5,5,6)]]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
xc_array=np.linspace(1.2,5.0,39)
molecule=lambda x: """H 0 0 0; F 0 0 %f"""%x
molecule_name=r"Hydrogen Fluoride"
print("CCSD")
energiesCC=CC_energy_curve(xc_array,basis,molecule=molecule)
energiesHF=energy_curve_RHF(xc_array,basis,molecule=molecule)
dicty={}
kvals=[1,2,4,6]
dicty["x"]=xc_array
dicty["HF"]=energiesHF
dicty["CCSD"]=energiesCC
dicty["kvals"]=kvals
dicty["sample_geometry"]=sample_geometry
energies_sample=[[],[]]
titles=[[],[]]
EVC_energies=[[[],[]],[[],[]]]
overlap_to_HF=np.zeros((len(kvals),len(xc_array)))
types=["procrustes","transform"]
name=["Gen. Procrustes","Symm. Orth."]
for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):
        sx=sample_geometry[i][j]
        for ki,k in enumerate(kvals):
            print("Eigvec (%d)"%(k))
            HF=eigvecsolver_RHF(sx[:k],basis,molecule=molecule,type=types[i])
            energiesEC,eigenvectors=HF.calculate_energies(xc_array)
            EVC_energies[i][j].append(energiesEC)
            axes[i,j].plot(xc_array,energiesEC,label="EC (%d pt.)"%(k))
            if i==1 and j==0:
                overlap_to_HF[ki,:]=HF.calculate_overlap_to_HF(xc_array)**2
        energiesHF_sample=energy_curve_RHF(sx,basis,molecule=molecule)
        axes[i,j].plot(sx,energiesHF_sample,"*",color="black",label="Sample points")
        axes[i,j].plot(xc_array,energiesHF,label="RHF")
        axes[i,j].plot(xc_array,energiesCC,label="CCSD")
        energies_sample[i].append(energiesHF_sample)
        titles[i].append(name[i])
        axes[i,j].set_title(name[i])
for ki,k in enumerate(kvals):
    print("K: %d"%k)
    for i,xc in enumerate(xc_array):
        print("xc=%.3f, overlap=%f"%(xc,overlap_to_HF[ki,i]))
dicty["energies_sample"]=energies_sample
dicty["titles"]=titles
dicty["EVC_energies"]=EVC_energies
file="energy_data/HF_data.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(dicty,f)



handles, labels = axes[0][0].get_legend_handles_labels()






axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("distance (Bohr)")
axes[1][1].set_xlabel("distance (Bohr)")
#axes[1][1].set_ylim([-100.4,-99.6])
fig.legend(handles, labels,loc="lower right")
fig.tight_layout()
fig.subplots_adjust(right=0.82)
plt.savefig("HF_EVC.pdf")
plt.show()
