import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import linalg
from pyscf import gto, scf, mcscf, fci, cc, mp,ao2mo
from guptri_py import *
from scipy.linalg import eig, qz, block_diag, eig, orth, fractional_matrix_power, expm
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment

import warnings
import sys
np.set_printoptions(linewidth=300,precision=3,suppress=True)
def localize_cholesky(mol,mo_coeffc,mo_occ):
    mo=cholesky_coefficientmatrix(mo_coeffc[:,mo_occ>0])
    mo=swappistan(mo)
    mo_coeffc[:,mo_occ>0]=np.array(mo)
    mo_unocc=cholesky_coefficientmatrix(mo_coeffc[:,mo_occ<=0])
    mo_unocc=swappistan(mo_unocc)
    mo_coeffc[:,mo_occ<=0]=np.array(mo_unocc)
    print(mo_coeffc)
    return mo_coeffc
np.set_printoptions(linewidth=300,precision=10,suppress=True)
def basischange(C_old,overlap_AOs_newnew,neh):
    C_old=C_old.copy()
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
def orthogonal_procrustes(mo_new,reference_mo,weights=None):
    A=mo_new
    B=reference_mo.copy()
    if weights is not None:
        B=B@np.diag(weights)
    M=A.T@B
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0
def localize_procrustes(mol,mo_coeff,mo_occ,ref_mo_coeff,mix_states=False,active_orbitals=None,nelec=None, return_R=False,weights=None):
    """Performs the orthgogonal procrustes on the occupied and the unoccupied molecular orbitals.
    ref_mo_coeff is the mo_coefs of the reference state.
    If "mix_states" is True, then mixing of occupied and unoccupied MO's is allowed.
    """
    if active_orbitals is None:
        active_orbitals=np.arange(len(mo_coeff))
    if nelec is None:
        nelec=int(np.sum(mo_occ))
    active_orbitals_occ=active_orbitals[:nelec//2]
    active_orbitals_unocc=active_orbitals[nelec//2:]
    mo_coeff_new=mo_coeff.copy()
    if mix_states==False:
        mo=mo_coeff[:,active_orbitals_occ]
        premo=ref_mo_coeff[:,active_orbitals_occ]
        R1,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R1
        mo_unocc=mo_coeff[:,active_orbitals_unocc]
        premo=ref_mo_coeff[:,active_orbitals_unocc]
        R2,scale=orthogonal_procrustes(mo_unocc,premo)
        mo_unocc=mo_unocc@R2


        mo_coeff_new[:,active_orbitals_occ]=np.array(mo)
        mo_coeff_new[:,active_orbitals_unocc]=np.array(mo_unocc)
        R=block_diag(R1,R2)
    elif mix_states==True:
        mo=mo_coeff[:,active_orbitals]
        premo=ref_mo_coeff[:,active_orbitals]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff_new[:,active_orbitals]=np.array(mo)

    if return_R:
        return mo_coeff_new,R
    else:
        return mo_coeff_new

def schur_lowestEigenValue(H,S):
    HH, SS, Q, Z = qz(H, S)
    for i in range(len(SS)):
        if np.abs(SS[i,i])<1e-9:
            SS[i,i]=1e10
    e=np.diag(HH)/np.diag(SS)
    idx = np.real(e).argsort()
    e = e[idx]
    print(e)
    return np.real(e[0])
def guptri_Eigenvalue(H,S):
    SS, HH, P, Q, kstr = guptri(H,S,zero=True)
    nonzero=np.where(abs(HH[0,:])>1e-5)[0][0]
    SS_reduced=SS[0:len(SS)-nonzero,nonzero:]
    HH_reduced=HH[0:len(SS)-nonzero,nonzero:]
    e=np.diag(SS_reduced)/np.diag(HH_reduced)
    idx = np.real(e).argsort()
    e = e[idx]
    return np.real(e[0])
    return kstr[0]
def similiarize_natural_orbitals(noons_ref,natorbs_ref,noons,natorbs,nelec,S,Sref):
    pairs_ref=[]
    pairs=[]
    #For each pair of natural orbitals, make it procrustes-similar to the reference
    i=0
    #Step 1: Find pairs and pair indices

    while i<len(noons_ref):
        if i+1==len(noons_ref):
            break
        if abs(np.log(noons_ref[i])-np.log(noons_ref[i+1]))<1e-7:
            pairs_ref.append((i,i+1))
            i+=2
        else:
            i+=1
    i=0
    while i<len(noons):
        if i+1==len(noons):
            break
        if abs(np.log(noons[i])-np.log(noons[i+1]))<1e-7:
            pairs.append((i,i+1))
            i+=2
        else:
            i+=1

    for i in range(len(pairs)):
        new_orbs,t=orthogonal_procrustes(natorbs[:,pairs[i]],natorbs_ref[:,pairs_ref[i]])
        natorbs[:,pairs[i]]=natorbs[:,pairs[i]]@new_orbs
    print(pairs_ref,len(pairs_ref))
    print(pairs,len(pairs))
    similarities=natorbs_ref.T@scipy.linalg.fractional_matrix_power(Sref,0.5)@scipy.linalg.fractional_matrix_power(S,0.5)@natorbs
    assignment = linear_sum_assignment(-np.abs(similarities))[1]
    signs=[]
    for i in range(len(similarities)):
        signs.append(np.sign(similarities[i,assignment[i]]))
    natorbs=natorbs[:,assignment]*np.array(signs)
    noons=noons[assignment]
    pairs=[]

    return noons, natorbs
