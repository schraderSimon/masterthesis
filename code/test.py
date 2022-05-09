import pyscf
from pyscf import gto, scf
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("./eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *
from scipy.linalg import orthogonal_procrustes, norm, svd
np.set_printoptions(linewidth=200,precision=10,suppress=True)

def make_mol(molecule,x,basis="6-31G*"):
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
def diagonal_energy(onebody,twobody):
    alpha=np.arange(5)
    beta=np.arange(5)
    onebody_alpha=np.sum(np.diag(onebody)[alpha])
    onebody_beta=np.sum(np.diag(onebody)[beta])
    energy2=0
    #Equation 2.175 from Szabo, Ostlund
    for a in alpha:
        for b in alpha:
            energy2+=twobody[a,a,b,b]
            energy2-=twobody[a,b,b,a]
    for a in alpha:
        for b in beta:
            energy2+=twobody[a,a,b,b]
    for a in beta:
        for b in alpha:
            energy2+=twobody[a,a,b,b]
    for a in beta:
        for b in beta:
            energy2+=twobody[a,a,b,b]
            energy2-=twobody[a,b,b,a]
    energy2*=0.5
    return onebody_alpha+onebody_beta+energy2
mol1=make_mol(molecule,1.2,"6-31G")
mf=scf.RHF(mol1)
mf.kernel()
mo_1=mf.mo_coeff[:,mf.mo_occ>0]
xc_array=np.linspace(1.2,4.5,20)
Energies=[]
reference_mo=mo_1
def orthogonal_own(mo_new,reference_mo):
    A=reference_mo.T
    B=mo_new.T
    M=B@A.T
    U,s,Vt=svd(M)
    return U@Vt, 0
for x in xc_array:
    mol=make_mol(molecule,x,"6-31G")
    mf=scf.RHF(mol)
    mf.kernel()
    mo_new=mf.mo_coeff[:,mf.mo_occ>0]
    before=norm(mo_new-reference_mo)
    R_own,scale=orthogonal_own(mo_new,reference_mo)
    R,scale=orthogonal_procrustes(mo_new,reference_mo)
    print(np.max(np.abs(R_own-R)))
    after=norm(mo_new@R-reference_mo+1e-5)

    print("Before: %f, after: %f"%(before,after))
plt.plot(xc_array,Energies)
plt.show()
