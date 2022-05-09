import sys
sys.path.append("../../eigenvectorcontinuation/")
import matplotlib
from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
basis="cc-pVDZ"
sample_geometry=[[np.linspace(0,0.5,6),[0,0.1,0.2,3.8,3.9,4.0]],[np.linspace(0,0.5,6),[0,0.1,0.2,3.8,3.9,4.0]]]
sample_indices=[[[0,1,2,3,4,5],[0,1,2,38,39,40]],[[0,1,2,3,4,5],[0,1,2,38,39,40]]]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
mos_1=[]
mos_2=[]
occdict2={"A1":6,"B1":0,"B2":0}
occdict3={"A1":4,"B1":2,"B2":0}
occdict1={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]
xc_array=np.linspace(0,4,81)
energies=np.zeros((len(xc_array),3))
E_FCI=np.zeros(len(xc_array))
if basis=="cc-pVDZ":
    E_FCI=[-15.83643, -15.836093, -15.834828, -15.832713, -15.82985, -15.826351, -15.822342, -15.817944, -15.813265, -15.808388, -15.80337, -15.798227, -15.792946, -15.787481, -15.781763, -15.77571, -15.769235, -15.762252, -15.75469, -15.746497, -15.737641, -15.728118, -15.717955, -15.707218, -15.696025, -15.684583, -15.673302, -15.663154, -15.656804, -15.658575, -15.667306, -15.67869, -15.690784, -15.702886, -15.714654, -15.72583, -15.736151, -15.745301, -15.752877, -15.758353, -15.761034]
nelec=4
sample_coeffmatrices_1=[]
sample_coeffmatrices_2=[]
ground_coeffmatrices=[]
exc_coeffmatrices=[]
ground_energies=[]
sample_coeffmatrices_min=[]
for k,x in enumerate(xc_array):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    energies_temp=np.zeros(3)
    for i in [0,1,2]:
        mf = scf.RHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        mo_coeff_temp.append(mf.mo_coeff)
        mo_en_temp.append(mf.mo_energy)
        energies_temp[i]=e
    es=np.argsort(energies_temp)
    ground_energies.append(energies_temp[es[0]])
    if basis=="cc-pVDZ":
        pass
    else:
        mfci=fci.FCI(mf)
        e=mfci.kernel()[0]
        E_FCI[k]=e
        print("Pos: %f, EFCI: %f"%(x,e))
    ground_coeffmatrices.append(mo_coeff_temp[es[0]])
    exc_coeffmatrices.append(mo_coeff_temp[es[1]])
    if x<2.5:
        for i in range(3):
            energies[k,i]=energies_temp[es[i]]
        sample_coeffmatrices_1.append(mo_coeff_temp[es[0]])
        sample_coeffmatrices_2.append(mo_coeff_temp[es[1]])

    else:
        for i in range(3):
            energies[k,i]=energies_temp[i]
        sample_coeffmatrices_1.append(mo_coeff_temp[0])
        sample_coeffmatrices_2.append(mo_coeff_temp[1])
kvals1=[3,6]
kvals2=[6,12]
kvals=[kvals1,kvals2]
data={}
data["Phi1_E"]=energies[:,0]
data["Phi2_E"]=energies[:,1]
data["xc_array"]=xc_array
data["FCI"]=E_FCI
"""
file="energy_data/BeH2_data_631G*.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(data,f)
sys.exit(1)
"""
energies_3=[[],[]]
energies_6=[[],[]]
geometry_energy_pair=[[],[]]
for i in range(2):
    for j in range(2):
        if i==0:
            coefficient_matrices=[ground_coeffmatrices[l] for l in sample_indices[i][j]]

        elif i==1:
            coefficient_matrices=[]
            for l in sample_indices[i][j]:
                coefficient_matrices=coefficient_matrices+[ground_coeffmatrices[l]]+[exc_coeffmatrices[l]]
        sx=sample_geometry[i][j]
        for ki,k in enumerate(kvals[i]):
            print("Eigvec (%d)"%(k))
            HF=eigvecsolver_RHF(sx[:k],basis,molecule=molecule,type="transform",coeff_matrices=coefficient_matrices[:k])
            energiesEC,eigenvectors=HF.calculate_energies(xc_array)
            axes[i,j].plot(xc_array,energiesEC,label="EC (%d pt.)"%(k))
            if ki==0:
                energies_3[i].append(energiesEC)
            elif ki==1:
                energies_6[i].append(energiesEC)
        axes[i,j].plot(xc_array,energies[:,0],label=r"RHF $|\Phi_1\rangle$")
        axes[i,j].plot(xc_array,energies[:,1],label=r"RHF $|\Phi_2\rangle$")
        axes[i,j].plot(xc_array,E_FCI,label="FCI")
        if i==0:
            energiesHF_sample=[ground_energies[k] for k in sample_indices[i][j]]
            axes[i,j].plot(sx,energiesHF_sample,"*",color="black",label="Sample pts.")
            geometry_energy_pair[i].append(list(zip(sx,energiesHF_sample)))
        if i==1:
            energiesHF_sample=[energies[k,0] for k in sample_indices[i][j]]
            axes[i,j].plot(sx,energiesHF_sample,"*",color="black",label="Sample pts.")
            energiesHF_sample2=[energies[k,1] for k in sample_indices[i][j]]
            axes[i,j].plot(sx,energiesHF_sample2,"*",color="black")
            geometry_energy_pair[i].append( list(zip( np.concatenate((sx,sx)) , np.concatenate((energiesHF_sample,energiesHF_sample2)) )) )
data["energy3"]=energies_3
data["energy6"]=energies_6
data["samples"]=geometry_energy_pair
file="energy_data/BeH2_data.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(data,f)


handles, labels = axes[0][0].get_legend_handles_labels()
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("x (Bohr)")
axes[1][1].set_xlabel("x (Bohr)")
#axes[1][1].set_ylim([-100.4,-99.6])
fig.legend(handles, labels,loc="center right")
fig.tight_layout()
fig.subplots_adjust(right=0.75)
plt.savefig("BeH2_EVC.pdf")
plt.show()
