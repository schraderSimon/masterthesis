import pyscf
from pyscf import gto, scf
import numpy as np
import sys
sys.path.append("./eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *

np.set_printoptions(linewidth=200,precision=5,suppress=True)

def make_mol(molecule,x,basis="6-31G"):
    mol=gto.Mole()
    mol.atom=molecule(x)
    mol.basis = basis
    mol.unit= "Bohr"
    mol.build()
    return mol
molecule=lambda x: "H 0 0 0; F 0 0 %f"%x
def basischange(C_old,overlap_AOs_newnew):
    S=np.einsum("mi,vj,mv->ij",C_old,C_old,overlap_AOs_newnew)
    S_eig,S_U=np.linalg.eigh(S)
    S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
    C_new=np.einsum("ij,mj->mi",S_powerminusonehalf,C_old)
    return C_new
def getdeterminant_matrix(AO_overlap,HF_coefficients_left,HF_coefficients_right):
    determinant_matrix=np.einsum("ab,ai,bj->ij",AO_overlap,HF_coefficients_left,HF_coefficients_right)
    return determinant_matrix
mol1=make_mol(molecule,1)
mf=scf.RHF(mol1)
mf.kernel()
mo_1=mf.mo_coeff[:,mf.mo_occ>0]

mol2=make_mol(molecule,5.0)
mf=scf.RHF(mol2)
mf.kernel()
mo_2=basischange(mf.mo_coeff[:,mf.mo_occ>0],mol1.intor("int1e_ovlp"))

S=getdeterminant_matrix(mol1.intor("int1e_ovlp"),mo_1,mo_2)
print(np.linalg.det(S))
Linv,D,Uinv=LDU_decomp(S)
L=Linv.T
R=Uinv.T
mo_1new=mo_1@Linv
mo_2new=mo_2@R
S=getdeterminant_matrix(mol1.intor("int1e_ovlp"),mo_1new,mo_2new)
print(1e-10+mo_1new@mo_1new.T-mo_1@mo_1.T)
print(1e-10+mo_2new@mo_2new.T-mo_2@mo_2.T)
