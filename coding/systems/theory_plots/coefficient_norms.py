import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf,cc
from scipy.linalg import block_diag, eig, orth
import copy
from numpy.linalg import norm
import sys
np.set_printoptions(linewidth=300,precision=10,suppress=True)

sys.path.append("../../eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *

def localize_cholesky(mol,mo_coeffc,mo_occ):
    mo=cholesky_coefficientmatrix(mo_coeffc[:,mo_occ>0])
    mo=swappistan(mo)
    mo_coeffc[:,mo_occ>0]=np.array(mo)
    mo_unocc=cholesky_coefficientmatrix(mo_coeffc[:,mo_occ<=0])
    mo_unocc=swappistan(mo_unocc)
    mo_coeffc[:,mo_occ<=0]=np.array(mo_unocc)
    print(mo_coeffc)
    return mo_coeffc

def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo.T
    B=mo_new.T
    M=B@A.T
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0
def localize_procrustes(mol,mo_coeff,mo_occ,ref_mo_coeff,mix_states=False):
    """Performs the orthgogonal procrustes on the occupied and the unoccupied molecular orbitals.
    ref_mo_coeff is the mo_coefs of the reference state.
    If "mix_states" is True, then mixing of occupied and unoccupied MO's is allowed.
    """
    if mix_states==False:
        mo=mo_coeff[:,mo_occ>0]
        premo=ref_mo_coeff[:,mo_occ>0]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff[:,mo_occ>0]=np.array(mo)
        mo_unocc=mo_coeff[:,mo_occ<=0]
        premo=ref_mo_coeff[:,mo_occ<=0]
        R,scale=orthogonal_procrustes(mo_unocc,premo)
        mo_unocc=mo_unocc@R

        mo_coeff[:,mo_occ<=0]=np.array(mo_unocc)


    elif mix_states==True:
        mo=mo_coeff[:,:]
        premo=ref_mo_coeff[:,:]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff[:,:]=np.array(mo)
    return mo_coeff
def basischange(C_old,overlap_AOs_newnew,neh):
    def overlap_p(L,R):
        return np.einsum("i,j,ij->",L,R,overlap_AOs_newnew)
    C_occ=C_old[:,:neh]

    S_occ=np.einsum("mi,vj,mv->ij",C_occ,C_occ,overlap_AOs_newnew)
    S_eig,S_U=np.linalg.eigh(S_occ)
    S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
    C_new_occ=np.einsum("ij,mj->mi",S_powerminusonehalf,C_occ)
    #Remove C_occ part from the unoccupied matrices...

    C_unocc=C_old[:,neh:]
    for unocc_col in range(C_unocc.shape[1]):
        for occ_col in range(C_new_occ.shape[1]):
            C_unocc[:,unocc_col]-=C_new_occ[:,occ_col]*overlap_p(C_new_occ[:,occ_col],C_unocc[:,unocc_col])
    S_unocc=np.einsum("mi,vj,mv->ij",C_unocc,C_unocc,overlap_AOs_newnew)
    S_eig,S_U=np.linalg.eigh(S_unocc)
    S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
    C_new_unocc=np.einsum("ij,mj->mi",S_powerminusonehalf,C_unocc)
    C_new=np.zeros_like(C_old)
    C_new[:,:neh]=C_new_occ
    C_new[:,neh:]=C_new_unocc
    return C_new
def make_mol(molecule,x,basis="6-31G"):
    mol=gto.Mole()
    mol.atom=molecule(x)
    mol.basis = basis
    mol.unit= "Bohr"
    mol.build()
    return mol
def get_reference_determinant(molecule_func,refx,basis,charge):
    mol = gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = False
    mol.build(atom=molecule_func(*refx), basis=basis)
    hf = scf.RHF(mol)
    hf.kernel()
    return np.asarray(localize_cholesky(mol,hf.mo_coeff,hf.mo_occ))

molecule=lambda x: "F 0 0 0; H 0 0 %f"%x
#molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
basis="6-31G"
molecule_name="HF"
x_sol=np.linspace(1.3,4,28)
#x_sol=np.array([1.9,2.5])
ref_x=[2]
mol=make_mol(molecule,ref_x[0],basis)
neh=mol.nelectron//2
mfref=scf.RHF(mol)
mfref.kernel()
ref_coefficientmatrix=localize_cholesky(mol,mfref.mo_coeff,mfref.mo_occ).copy()
mfref.mo_coeff=ref_coefficientmatrix
mycc=cc.CCSD(mfref); mycc.kernel(); ref_t2=mycc.t2

norms_coefficientmatrix=np.zeros((3,len(x_sol)))
norms_T2=np.zeros((3,len(x_sol)))
for i,x in enumerate(x_sol):
    mol=make_mol(molecule,x,basis)
    mfnew=scf.RHF(mol)
    mfnew.kernel()
    cholesky_coeff=localize_cholesky(mol,mfnew.mo_coeff,mfnew.mo_occ).copy()
    transformed_coeff=basischange(ref_coefficientmatrix,mol.intor("int1e_ovlp"),neh).copy()
    procrustes_coeff=localize_procrustes(mol,mfnew.mo_coeff,mfnew.mo_occ,ref_coefficientmatrix,mix_states=False).copy()
    mf_cholesky=scf.RHF(mol); mf_cholesky.kernel(); mf_cholesky.mo_coeff=cholesky_coeff
    mf_transformed=scf.RHF(mol); mf_transformed.kernel();  mf_transformed.mo_coeff=transformed_coeff
    mf_procrustes=scf.RHF(mol); mf_procrustes.kernel();  mf_procrustes.mo_coeff=procrustes_coeff
    mycc=cc.CCSD(mf_procrustes); mycc.kernel(); procrustes_t2=mycc.t2
    mycc=cc.CCSD(mf_transformed); mycc.kernel(); transformed_t2=mycc.t2
    mycc=cc.CCSD(mf_cholesky); mycc.kernel(); cholesky_t2=mycc.t2
    norms_coefficientmatrix[0,i]=norm(procrustes_coeff-ref_coefficientmatrix)
    norms_coefficientmatrix[1,i]=norm(transformed_coeff-ref_coefficientmatrix)
    norms_coefficientmatrix[2,i]=norm(cholesky_coeff-ref_coefficientmatrix)
    norms_T2[0,i]=norm(procrustes_t2-ref_t2)
    norms_T2[1,i]=norm(transformed_t2-ref_t2)
    norms_T2[2,i]=norm(cholesky_t2-ref_t2)
    #print(cholesky_coeff-ref_coefficientmatrix)
labels=["Procrustes orbitals", "Converted orbitals", "Choleksy orbitals"]
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.set_title(r"$||C(x)-C(x_{{ref}})||$")
ax2.set_title(r"$||T_2(x)-T_2(x_{{ref}})||$")
for i in range(3):
    ax1.plot(x_sol,norms_coefficientmatrix[i,:],label=labels[i])
    ax2.plot(x_sol,norms_T2[i,:],label=labels[i])
ax1.legend()
ax2.legend()
ax2.set_xlabel("Interatomic distance (Bohr)")
ax1.set_xlabel("Interatomic distance (Bohr)")
plt.tight_layout()
plt.savefig("HF_coefficient_norms.pdf")
plt.show()
