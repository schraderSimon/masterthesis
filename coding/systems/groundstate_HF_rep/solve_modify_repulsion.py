import sys
sys.path.append("../../eigenvectorcontinuation/")

from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    fig, ax = plt.subplots(figsize=(9,6))

    basis="cc-pVDZ"
    def molecule(x):
        return "F 0 0 0; H 0 0 %f"%x
    molecule_name=r"Hydrogen Fluoride"
    xc_array=np.linspace(1.2,5,49)
    '''
    basis="6-31G*"
    molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
    molecule_name="BeH2"
    xc_array=np.linspace(0,4,41)
    '''
    energies_HF=energy_curve_RHF(xc_array,basis,molecule=molecule)
    sample_strengths=np.linspace(1,0,11)
    additions=np.array([-0.1,1.1])
    sample_strengths=np.concatenate((sample_strengths,additions))
    for i in range(13,1,-3):
        print("Eigvec (%d)"%(i))
        HF=eigvecsolver_RHF_coupling(sample_strengths[:i],xc_array,basis,molecule=molecule,symmetry=True)
        energiesEC,eigenvectors=HF.calculate_energies(xc_array)
        print(energiesEC)
        ax.plot(xc_array,energiesEC,label="EC (%d points), %s"%(i,basis))
    ax.plot(xc_array,energies_HF,label="RHF,%s"%basis)
    ax.plot(xc_array,CC_energy_curve(xc_array,basis,molecule=molecule),label="CCSD(T),%s"%basis)

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