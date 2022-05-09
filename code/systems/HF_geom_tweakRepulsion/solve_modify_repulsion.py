import sys
sys.path.append("../libraries")

from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
if __name__=="__main__":
    fig, ax = plt.subplots(figsize=(9,6))

    basis="6-31G*"
    def molecule(x):
        return "F 0 0 0; H 0 0 %f"%x
    molecule_name=r"Hydrogen Fluoride"
    xc_array=np.linspace(1.2,5,39)
    #xc_array=np.linspace(3.2,3.7,6)


    #xc_array=np.linspace(0,4,41)
    energies_HF=energy_curve_RHF(xc_array,basis,molecule=molecule)
    sample_strengths=np.linspace(1.1,0.5,13)
    #additions=np.array([-0.1,1.1])
    #sample_strengths=np.concatenate((sample_strengths,additions))
    data={}
    data["x"]=xc_array
    data["strenghts"]=sample_strengths
    data["RHF"]=energies_HF

    for i in range(13,1,-3):
        print("Eigvec (%d)"%(i))
        HF=eigvecsolver_RHF_coupling(sample_strengths[:i],xc_array,basis,molecule=molecule,symmetry=True)
        energiesEC,eigenvectors=HF.calculate_energies(xc_array)
        print(energiesEC)
        data["%d"%i]=energiesEC
        ax.plot(xc_array,energiesEC,label="EC (%d points)"%(i))
    ax.plot(xc_array,energies_HF,label="RHF,%s"%basis)
    CC_energy=CC_energy_curve(xc_array,basis,molecule=molecule)
    ax.plot(xc_array,CC_energy,label="CCSD,%s"%basis)
    data["CC"]=CC_energy
    string=r"c $\in$["
    for xc in sample_strengths:
        string+="%.1f,"%xc
    string=string[:-1]
    string+="]"
    ax.text(0.5, 0.9, string, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    ax.set_title("Potential energy curve for %s"%molecule_name)
    ax.set_xlabel("Atomic distance (Bohr)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_ylim([-100.3,-99.4])
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig("repulsion_%s.pdf"%molecule_name)
    plt.show()

    if basis=="6-31G*":
        basis="6-31Gstjerne"
    file="HF_data_%s.bin"%basis
    import pickle
    with open(file,"wb") as f:
        pickle.dump(data,f)
