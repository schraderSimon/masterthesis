"""
Contains the functions to perform eigenvector continuation at a single molecular position R
for a set of eigenfunctions that are centered around a position R, where the parameter $\lambda$ is the electron-electron repulsion.
This can improve the RHF result, but does not seem to converge (close to) the full CI limit.
"""
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../eigenvectorcontinuation_molecule')
import scipy as sp
import numpy as np
import pyscf
from pyscf import gto, scf, ci, cc, ao2mo

from ec_HF_basischange import *
from eigenvectorcontinuation import generalized_eigenvector
helium=gto.Mole()
helium.atom="He 0 0 0"
helium.basis="6-31G"
helium.unit="AU"
helium.build()

mf = scf.RHF(helium)
#print(helium.intor('int2e',aosym="s1"))
mf.kernel()
eri=helium.intor('int2e',aosym="s1")*1
mf._eri = ao2mo.restore(4,eri,2) #update matrix
helium.incore_anyway=True
print(mf._eri)
mf.kernel()
class eigvecsolver_RHF_coupling(eigvecsolver_RHF):
    def __init__(self,sample_lambdas,sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False):
        self.sample_positions=sample_points
        self.HF_coefficients=[] #The Hartree Fock coefficients solved at the sample points
        self.molecule=molecule
        self.basis_type=basis_type
        self.spin=spin
        self.unit=unit
        self.charge=charge
        self.symmetry=symmetry
        self.sample_points=sample_lambdas
    def solve_HF(self,sample_point):
        """Solve equations for different RHF's"""
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(sample_point)
            mf = scf.RHF(mol)
            eri=mol.intor('int2e',aosym="s1")*x
            mf._eri = ao2mo.restore(1,eri,mol.nao_nr())
            mol.incore_anyway=True
            mf.kernel()
            expansion_coefficients_mol= mf.mo_coeff[:, mf.mo_occ > 0.]
            HF_coefficients.append(expansion_coefficients_mol)
        self.HF_coefficients=HF_coefficients
    def calculate_energies(self,xc_array):
            """Calculates the molecule's energy"""
            energy_array=np.zeros(len(xc_array))
            eigval_array=[]
            for index,xc in enumerate(xc_array):
                self.solve_HF(xc) #Update HF coefficients
                mol_xc=self.build_molecule(xc)
                new_HF_coefficients=self.HF_coefficients #No need to basis change (same basis)
                S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
                try:
                    eigval,eigvec=generalized_eigenvector(T,S)
                except:
                    eigval=float('NaN')
                    eigvec=float('NaN')
                energy_array[index]=eigval
                eigval_array.append(eigval)
            return energy_array,eigval_array
class eigvecsolver_UHF_coupling(eigvecsolver_UHF):
    def __init__(self,sample_lambdas,sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False):
        self.sample_positions=sample_points
        self.HF_coefficients=[] #The Hartree Fock coefficients solved at the sample points
        self.molecule=molecule
        self.basis_type=basis_type
        self.spin=spin
        self.unit=unit
        self.charge=charge
        self.symmetry=symmetry
        self.sample_points=sample_lambdas
    def solve_HF(self,sample_point):
        """Solve equations for different RHF's"""
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(sample_point)
            mf = scf.UHF(mol)
            eri=mol.intor('int2e',aosym="s1")*x
            mf._eri = ao2mo.restore(1,eri,mol.nao_nr())
            mol.incore_anyway=True
            dm_alpha, dm_beta = mf.get_init_guess()
            dm_beta[:,:] = 0
            dm = (dm_alpha,dm_beta)
            mf.kernel(dm)
            expansion_coefficients_mol_alpha=mf.mo_coeff[0][:, mf.mo_occ[0] > 0.]
            expansion_coefficients_mol_beta =mf.mo_coeff[1][:, mf.mo_occ[1] > 0.]
            HF_coefficients.append([expansion_coefficients_mol_alpha,expansion_coefficients_mol_beta])
        self.HF_coefficients=HF_coefficients
    def calculate_energies(self,xc_array):
            """Calculates the molecule's energy"""
            energy_array=np.zeros(len(xc_array))
            eigval_array=[]
            for index,xc in enumerate(xc_array):
                self.solve_HF(xc) #Update HF coefficients
                mol_xc=self.build_molecule(xc)
                new_HF_coefficients=self.HF_coefficients #No need to basis change (same basis)
                S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
                try:
                    eigval,eigvec=generalized_eigenvector(T,S)
                except:
                    eigval=float('NaN')
                    eigvec=float('NaN')
                energy_array[index]=eigval
                eigval_array.append(eigval)
            return energy_array,eigval_array

def RHF_energy_e2strength(strengths,basis_type,molecule):
    energies=[]
    for i,strength in enumerate(strengths):
        mol1=gto.Mole()
        mol1.atom=molecule(0) #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit='AU'
        mol1.spin=0 #Assume closed shell
        mol1.verbose=2
        mol1.build()
        mf=scf.RHF(mol1)
        eri=mol1.intor('int2e')*strength
        mf._eri = ao2mo.restore(1,eri,mol1.nao_nr())
        #mol1.incore_anyway=True
        energy=mf.kernel()
        energies.append(energy)
    return np.array(energies)
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
