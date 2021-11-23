import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo,fci
import sys
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.linalg import block_diag, eig, orth
from numba import jit
from matrix_operations import *
from helper_functions import *
from scipy.optimize import minimize, root,newton
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
        #print(mo_unocc)


    elif mix_states==True:
        mo=mo_coeff[:,:]
        premo=ref_mo_coeff[:,:]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff[:,:]=np.array(mo)
    return mo_coeff
def make_mol(molecule,x,basis="6-31G",charge=0):
	mol=gto.Mole()
	mol.atom=molecule(x)
	mol.basis = basis
	mol.unit= "Bohr"
	mol.charge=charge
	mol.build()
	return mol
#molecule=lambda x: "H 0 0 0; F 0 0 %f"%x
molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))

def tau_mat(ts,td,Nelec,dim):
	ts=ts
	td=td
	tau_rel=td.copy()
	tau_rel=tau_rel+np.einsum("ai,bj->abij",ts,ts)
	tau_rel=tau_rel-np.einsum("bi,aj->abij",ts,ts)
	return tau_rel
def taus_mat(ts,td,Nelec,dim):
	ts=ts
	td=td
	tausm_rel=td.copy()
	tausm_rel=tausm_rel+np.einsum("ai,bj->abij",ts,ts)*0.5
	tausm_rel=tausm_rel-np.einsum("bi,aj->abij",ts,ts)*0.5
	return tausm_rel
def updateintermediates(ts,td,Nelec,dim,fs,spinints,x=True):
	# Stanton eq (3)
	tau=tau_mat(ts,td,Nelec,dim)
	taus=taus_mat(ts,td,Nelec,dim)
	ts=ts
	td=td
	F = np.zeros((dim,dim))
	F[Nelec:dim,Nelec:dim]=fs[Nelec:dim,Nelec:dim]
	np.fill_diagonal(F[Nelec:dim,Nelec:dim],0)
	W = np.zeros((dim,dim,dim,dim))
	F[Nelec:dim,Nelec:dim]-=0.5*np.einsum("me,am->ae",fs[:Nelec,Nelec:dim],ts,optimize=True)
	F[Nelec:dim,Nelec:dim]+=np.einsum("fm,mafe->ae",ts,spinints[:Nelec,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	F[Nelec:dim,Nelec:dim]-=0.5*np.einsum("afmn,mnef->ae",taus,spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	# Stanton eq (4)
	F[:Nelec,:Nelec]=fs[:Nelec,:Nelec]
	np.fill_diagonal(F[:Nelec,:Nelec],0)
	F[:Nelec,:Nelec]+=0.5*np.einsum("ei,me->mi",ts,fs[:Nelec,Nelec:dim],optimize=True)
	F[:Nelec,:Nelec]+=np.einsum("en,mnie->mi",ts,spinints[:Nelec,:Nelec,:Nelec,Nelec:dim],optimize=True)
	F[:Nelec,:Nelec]+=0.5*np.einsum("efin,mnef->mi",taus,spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (5)
	F[:Nelec,Nelec:dim]=fs[:Nelec,Nelec:dim]
	F[:Nelec,Nelec:dim]+=np.einsum("fn,mnef->me",ts,spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (6)
	W[:Nelec,:Nelec,:Nelec,:Nelec]=spinints[:Nelec,:Nelec,:Nelec,:Nelec]
	W[:Nelec,:Nelec,:Nelec,:Nelec]+=np.einsum("ej,mnie->mnij",ts,spinints[:Nelec,:Nelec,:Nelec,Nelec:dim],optimize=True)
	W[:Nelec,:Nelec,:Nelec,:Nelec]-=np.einsum("ei,mnje->mnij",ts,spinints[:Nelec,:Nelec,:Nelec,Nelec:dim],optimize=True)
	W[:Nelec,:Nelec,:Nelec,:Nelec]+=0.25*np.einsum("efij,mnef->mnij",tau,spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (7)
	W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim] = spinints[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]
	W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]+=np.einsum("am,bmef->abef",ts,spinints[Nelec:dim,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]-=np.einsum("bm,amef->abef",ts,spinints[Nelec:dim,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]+=0.25*np.einsum("abmn,mnef->abef",tau,spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (8)
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec] = spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec]
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec]+=np.einsum("fj,mbef->mbej",ts,spinints[:Nelec,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec]-=np.einsum("bn,mnej->mbej",ts,spinints[:Nelec,:Nelec:,Nelec:dim,:Nelec],optimize=True)
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec]-=0.5*np.einsum("fbjn,mnef->mbej",td,spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec]-=np.einsum("fj,bn,mnef->mbej",ts,ts,spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	return F,W
def T1_eq(ts,td,fs,spinints,F,Nelec,dim,Dai):
	ts=ts
	td=td
	tsnew=np.einsum("ai->ia",fs[:Nelec,Nelec:dim],optimize=True)
	tsnew=tsnew+np.einsum("ei,ae->ai",ts,F[Nelec:dim,Nelec:dim],optimize=True)
	tsnew=tsnew-np.einsum("am,mi->ai",ts,F[:Nelec,:Nelec],optimize=True)
	tsnew=tsnew+np.einsum("aeim,me->ai",td,F[:Nelec,Nelec:dim],optimize=True)
	tsnew=tsnew-0.5*np.einsum("efim,maef->ai",td,spinints[:Nelec,Nelec:,Nelec:,Nelec:],optimize=True)
	tsnew=tsnew-0.5*np.einsum("aemn,nmei->ai",td,spinints[:Nelec,:Nelec,Nelec:dim,:Nelec],optimize=True)
	tsnew=tsnew-np.einsum("fn,naif->ai",ts,spinints[:Nelec,Nelec:dim,:Nelec,Nelec:dim],optimize=True)
	tsnew=tsnew-np.einsum("ai,ai->ai",ts,Dai[Nelec:dim,:Nelec],optimize=True)
	return tsnew

def T2_eq(ts,td,fs,spinints,F,Nelec,dim,Dabij,W):
	tau=tau_mat(ts,td,Nelec,dim)
	tdnew=np.zeros((dim-Nelec,dim-Nelec,Nelec,Nelec))
	tdnew=np.einsum("abij->ijab",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	tdnew=tdnew+np.einsum("aeij,be->abij",td,F[Nelec:dim,Nelec:dim],optimize=True)
	tdnew=tdnew-np.einsum("beij,ae->abij",td,F[Nelec:dim,Nelec:dim],optimize=True)
	tdnew=tdnew-0.5*np.einsum("aeij,bm,me->abij",td,ts,F[:Nelec,Nelec:dim],optimize=True)
	tdnew=tdnew+0.5*np.einsum("beij,am,me->abij",td,ts,F[:Nelec,Nelec:dim],optimize=True)
	tdnew=tdnew-0.5*np.einsum("abim,ej,me->abij",td,ts,F[:Nelec,Nelec:dim],optimize=True)
	tdnew=tdnew+0.5*np.einsum("abjm,ei,me->abij",td,ts,F[:Nelec,Nelec:dim],optimize=True)
	tdnew=tdnew-np.einsum("abim,mj->abij",td,F[:Nelec,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("abjm,mi->abij",td,F[:Nelec,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("ei,abej->abij",ts,spinints[Nelec:dim,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("ej,abei->abij",ts,spinints[Nelec:dim,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+0.5*np.einsum("efij,abef->abij",tau,W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	tdnew=tdnew-np.einsum("am,mbij->abij",ts,spinints[:Nelec,Nelec:dim,:Nelec,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("bm,maij->abij",ts,spinints[:Nelec,Nelec:dim,:Nelec,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("aeim,mbej->abij",td,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("ei,am,mbej->abij",ts,ts,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("aejm,mbei->abij",td,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("ej,am,mbei->abij",ts,ts,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("beim,maej->abij",td,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("ei,bm,maej->abij",ts,ts,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("bejm,maei->abij",td,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("ej,bm,maei->abij",ts,ts,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+0.5*np.einsum("abmn,mnij->abij",tau,W[:Nelec,:Nelec,:Nelec,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("abij,abij->abij",td,Dabij[Nelec:dim,Nelec:dim,:Nelec,:Nelec],optimize=True)
	return tdnew
def makeT1(ts,td,fs,spinints,F,Nelec,dim,Dai,x=True):

	tsnew=np.zeros((dim-Nelec,Nelec))
	tsnew=np.einsum("ai->ia",fs[:Nelec,Nelec:dim],optimize=True)
	tsnew=tsnew+np.einsum("ei,ae->ai",ts,F[Nelec:dim,Nelec:dim],optimize=True)
	tsnew=tsnew-np.einsum("am,mi->ai",ts,F[:Nelec,:Nelec],optimize=True)
	tsnew=tsnew+np.einsum("aeim,me->ai",td,F[:Nelec,Nelec:dim],optimize=True)
	tsnew=tsnew-0.5*np.einsum("efim,maef->ai",td,spinints[:Nelec,Nelec:,Nelec:,Nelec:],optimize=True)
	tsnew=tsnew-0.5*np.einsum("aemn,nmei->ai",td,spinints[:Nelec,:Nelec,Nelec:dim,:Nelec],optimize=True)
	tsnew=tsnew-np.einsum("fn,naif->ai",ts,spinints[:Nelec,Nelec:dim,:Nelec,Nelec:dim],optimize=True)
	tsnew=tsnew/Dai[Nelec:dim,:Nelec]
	tsnew=tsnew
	return tsnew
# Stanton eq (2)
def makeT2(ts,td,fs,spinints,F,Nelec,dim,Dabij,W):
	tau=tau_mat(ts,td,Nelec,dim)
	tdnew=np.zeros((dim-Nelec,dim-Nelec,Nelec,Nelec))
	tdnew=np.einsum("abij->ijab",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	tdnew=tdnew+np.einsum("aeij,be->abij",td,F[Nelec:dim,Nelec:dim],optimize=True)
	tdnew=tdnew-np.einsum("beij,ae->abij",td,F[Nelec:dim,Nelec:dim],optimize=True)
	tdnew=tdnew-0.5*np.einsum("aeij,bm,me->abij",td,ts,F[:Nelec,Nelec:dim],optimize=True)
	tdnew=tdnew+0.5*np.einsum("beij,am,me->abij",td,ts,F[:Nelec,Nelec:dim],optimize=True)
	tdnew=tdnew-0.5*np.einsum("abim,ej,me->abij",td,ts,F[:Nelec,Nelec:dim],optimize=True)
	tdnew=tdnew+0.5*np.einsum("abjm,ei,me->abij",td,ts,F[:Nelec,Nelec:dim],optimize=True)
	tdnew=tdnew-np.einsum("abim,mj->abij",td,F[:Nelec,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("abjm,mi->abij",td,F[:Nelec,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("ei,abej->abij",ts,spinints[Nelec:dim,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("ej,abei->abij",ts,spinints[Nelec:dim,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+0.5*np.einsum("efij,abef->abij",tau,W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	tdnew=tdnew-np.einsum("am,mbij->abij",ts,spinints[:Nelec,Nelec:dim,:Nelec,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("bm,maij->abij",ts,spinints[:Nelec,Nelec:dim,:Nelec,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("aeim,mbej->abij",td,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("ei,am,mbej->abij",ts,ts,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("aejm,mbei->abij",td,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("ej,am,mbei->abij",ts,ts,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("beim,maej->abij",td,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("ei,bm,maej->abij",ts,ts,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+np.einsum("bejm,maei->abij",td,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew-np.einsum("ej,bm,maei->abij",ts,ts,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew=tdnew+0.5*np.einsum("abmn,mnij->abij",tau,W[:Nelec,:Nelec,:Nelec,:Nelec],optimize=True)
	tdnew=tdnew/Dabij[Nelec:dim,Nelec:dim,:Nelec,:Nelec]
	return tdnew
# Expression from Crawford, Schaefer (2000)
# DOI: 10.1002/9780470125915.ch2
# Equation (134) and (173)
# computes CCSD energy given T1 and T2

def ccsdenergy(fs,spinints,ts,td,Nelec,dim):
	ECCSD = np.einsum("ia,ai->",fs[:Nelec,Nelec:dim],ts,optimize=True)
	ECCSD=ECCSD+0.25*np.einsum("ijab,abij->",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],td,optimize=True)
	ECCSD=ECCSD+0.5*np.einsum("ijab,ai,bj->",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],ts,ts,optimize=True)
	return ECCSD
@jit(parallel=True,nopython=True)
def make_spinints_aokjemi(dim,energy_basis_2e_mol_chem,alternating):
    spinints_AO_kjemi=np.zeros((dim,dim,dim,dim))
    for p in range(dim):
    	spinP=int(2*(-(p%2)+0.5))
    	for q in range(dim):
    		spinQ=int(2*(-(q%2)+0.5))
    		for r in range(dim):
    			spinR=int(2*(-(r%2)+0.5))
    			for s in range(dim):
    				spinS=int(2*(-(s%2)+0.5))
    				P=p//2
    				Q=q//2
    				R=r//2
    				S=s//2
    				spinints_AO_kjemi[p,q,r,s]=energy_basis_2e_mol_chem[P,Q,R,S]*alternating[p,q]*alternating[r,s]
    return spinints_AO_kjemi
def best_CC_fit(t1s,t2s,eigvec):
    """Find the 'best' t1, t2 based on a linear CC-state"""
    def cost_function(sol,t1s,t2s,eigvec):

        num=len(eigvec)
        first_fitterino=np.zeros_like(t1s[0])
        second_fitterino=np.zeros_like(t2s[0])
        for i in range(num):
            first_fitterino=first_fitterino+t1s[i]*eigvec[i]
            second_fitterino=second_fitterino+t2s[i]*eigvec[i]*0.25
        for i in range(num):
            second_fitterino=second_fitterino+np.einsum("ai,bj->abij",t1s[i],t1s[i])*eigvec[i]*0.5
        a,i=t1s[0].shape
        t1_sol=np.reshape(sol[:a*i],(a,i))
        t2_sol=np.reshape(sol[a*i:],(a,a,i,i))
        first_sol=np.zeros_like(t1s[0])
        second_sol=np.zeros_like(t2s[0])
        first_sol=t1_sol
        second_sol=t2_sol*0.25
        second_sol=second_sol+np.einsum("ai,bj->abij",t1_sol,t1_sol)*0.5
        cost=np.sum((first_sol-first_fitterino)**2)+np.sum((second_sol-second_fitterino)**2)
        return cost
    def gradient(sol,t1s,t2s,eigvec):
        num=len(eigvec)
        first_fitterino=np.zeros_like(t1s[0])
        second_fitterino=np.zeros_like(t2s[0])
        for i in range(num):
            first_fitterino=first_fitterino+t1s[i]*eigvec[i]
            second_fitterino=second_fitterino+t2s[i]*eigvec[i]*0.25
        for i in range(num):
            second_fitterino=second_fitterino+np.einsum("ai,bj->abij",t1s[i],t1s[i])*eigvec[i]*0.5
        a,i=t1s[0].shape
        t1_grad=np.zeros((a,i))
        t2_grad=np.zeros((a,a,i,i))
        t1_sol=np.reshape(sol[:a*i],(a,i))
        t2_sol=np.reshape(sol[a*i:],(a,a,i,i))
        first_sol=np.zeros_like(t1s[0])
        second_sol=np.zeros_like(t2s[0])
        first_sol=t1_sol
        second_sol=t2_sol*0.25
        second_sol=second_sol+np.einsum("ai,bj->abij",t1_sol,t1_sol)*0.5
        for ass in range(a):
            for iss in range(i):
                t1_grad[ass,iss]=-2*(first_fitterino[ass,iss]-t1_sol[ass,iss])-2*np.einsum("bj,bj->",first_sol,second_fitterino[ass,:,iss,:]-second_sol[ass,:,iss,:])
        t2_grad=-0.5*(second_fitterino-second_sol)
        grad=np.concatenate((t1_grad,t2_grad),axis=None)
        return grad
    a,i=t1s[0].shape
    eigvec=np.real(eigvec)
    num=len(eigvec)
    t1_guess=np.zeros_like(t1s[0])
    t2_guess=np.zeros_like(t2s[0])
    for l in range(num):
        t1_guess=t1_guess+t1s[l]*eigvec[l]
        t2_guess=t2_guess+t2s[l]*eigvec[l]
    x0=np.concatenate((t1_guess,t2_guess),axis=None)
    x_sol=x0
    options={}
    options["gtol"]=1e-12
    options["ftol"]=1e-12
    res=minimize(cost_function,x0,args=(t1s,t2s,eigvec),jac=gradient,method="L-BFGS-B",options=options,tol=1e-12)
    x_sol=res.x
    success=res.success
    print("Success: ",success)
    print("Max gradient:",np.max(np.abs(gradient(x_sol,t1s,t2s,eigvec))))
    t1_new=np.reshape(x_sol[:a*i],(a,i))
    t2_new=np.reshape(x_sol[a*i:],(a,a,i,i))
    return t1_new,t2_new
#Step 1: Produce Fock matrix
mix_states=True
basis="6-31G"
molecule_name="N2"
ref_x=1.5
mol=make_mol(molecule,ref_x,basis,charge=0)
ENUC=mol.energy_nuc()
Nelec=mol.nelectron

print(ENUC)
mf=scf.RHF(mol)
ESCF=mf.kernel()
rhf_mo_ref=mf.mo_coeff

xnew=1.9
mol=make_mol(molecule,xnew,basis,charge=0)
mf=scf.RHF(mol)
ESCF=mf.kernel()
rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_mo_coeff=rhf_mo_ref,mix_states=mix_states)
#rhf_mo=mf.mo_coeff


mf.mo_coeff=rhf_mo
overlap_basis=block_diag(mol.intor("int1e_ovlp"),mol.intor("int1e_ovlp"))
energy_basis_1e=block_diag(mol.intor("int1e_kin")+mol.intor("int1e_nuc"),mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
gmf=scf.addons.convert_to_ghf(mf)


energy_basis_2e=mol.intor('int2e')

mycc = cc.CCSD(gmf)
mycc.kernel()
l1_pyscf,l2_pyscf=mycc.solve_lambda()

t1_pyscf=mycc.t1

mo_coeff=gmf.mo_coeff
fs=mo_coeff.T@gmf.get_fock()@mo_coeff #Fock matrix
print(fs)

energy_basis_1eMO = np.einsum('pi,pq,qj->ij', mo_coeff, energy_basis_1e, mo_coeff)
dim=len(energy_basis_2e)*2
dimhalf=len(energy_basis_2e)
energy_basis_2e_mol_chem=ao2mo.get_mo_eri(energy_basis_2e,(rhf_mo,rhf_mo,rhf_mo,rhf_mo)) #Molecular orbitals in spatial basis, not spin basis. Chemists notation

alternating=np.array([[(i+j)%2 for i in range(1,dim+1)] for j in range(dim)])
spinints_AO_kjemi=make_spinints_aokjemi(dim,energy_basis_2e_mol_chem,alternating)
spinints_AO_fysikk=np.transpose(spinints_AO_kjemi,(0,2,1,3))
spinints_AO_fysikk_antisymm=spinints_AO_fysikk-np.transpose(spinints_AO_fysikk,(0,1,3,2))
spinints=spinints_AO_fysikk_antisymm
def RHF_energy_nonconverged(Nelec2,energy_basis1,energy_basis_2e_mol,Enuc):
    E=0
    for i in range(Nelec2):
        E+=energy_basis1[i,i]
    E*=2
    for i in range(Nelec2):
        for j in range(Nelec2):
            E+=2*energy_basis_2e_mol[i,i,j,j]-energy_basis_2e_mol[i,j,j,i]
    return E+Enuc
def test_HF_energy():
    E=0
    for i in range(Nelec):
        E+=energy_basis_1eMO[i,i]
    for i in range(Nelec):
        for j in range(Nelec):
            E+=0.5*spinints[i,j,i,j]#energy_basis_2e_mol[i,i,j,j]-energy_basis_2e_mol[i,j,j,i]
    print(E+ENUC) #This should be the Hartree-Fock energy <3
test_HF_energy()

Dai = np.zeros((dim,dim))
for a in range(Nelec,dim):
	for i in range(0,Nelec):
		Dai[a,i] = fs[i,i] - fs[a,a]

# Stanton eq (13)
Dabij = np.zeros((dim,dim,dim,dim))
for a in range(Nelec,dim):
	for b in range(Nelec,dim):
		for i in range(0,Nelec):
			for j in range(0,Nelec):
				Dabij[a,b,i,j] = fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b]

ts = np.zeros_like(mycc.t1.T)
td = np.zeros_like(mycc.t2.T)

# Initial guess T2 --- from MP2 calculation!

for a in range(0,dim-Nelec):
	for b in range(0,dim-Nelec):
		for i in range(0,Nelec):
			for j in range(0,Nelec):
				td[a,b,i,j] += spinints[i,j,a+Nelec,b+Nelec]/(fs[i,i] + fs[j,j] - fs[a+Nelec,a+Nelec] - fs[b+Nelec,b+Nelec])
				pass

ECCSD = 0
DECC = 1
counter=0

while DECC > 1e-8: # arbitrary convergence criteria
	OLDCC = ECCSD
	F,W = updateintermediates(ts,td,Nelec,dim,fs,spinints)
	tsnew = makeT1(ts,td,fs,spinints,F,Nelec,dim,Dai)
	tdnew = makeT2(ts,td,fs,spinints,F,Nelec,dim,Dabij,W)
	ts = tsnew
	td = tdnew
	ECCSD = ccsdenergy(fs,spinints,ts,td,Nelec,dim)
	#print(DECC)
	DECC = abs(ECCSD - OLDCC)
	t1_val=T1_eq(ts,td,fs,spinints,F,Nelec,dim,Dai)
	t2_val=T2_eq(ts,td,fs,spinints,F,Nelec,dim,Dabij,W)
	print("Max T2-error",np.max(abs(t2_val)))
	print("Max T1-error",np.max(abs(t1_val)))
	print(DECC)
	counter+=1
print("E(corr,CCSD) = ", ECCSD)
print("E(CCSD) = ", ECCSD + ESCF)

print("Number of convergence steps: %d"%counter)
print("Difference in T1 coefficients compared to scipy: %e"%(np.max(np.abs(ts.T-mycc.t1))))
print("Difference in T2 coefficients compared to scipy: %e"%(np.max(np.abs(td.T-mycc.t2))))

#Now that the CC solver is in place, let's attempt to create functions for the calculation of the matrix elements as given by Ekstrøm & Hagen!!
#I will here only continue a single CC state, but this does not really affect the mechanics.
sample_x=np.linspace(1.5,2.5,11)
sample_energies=[]
t1s=[]
t2s=[]
l1s=[]
l2s=[]
for x in sample_x:
    mol=make_mol(molecule,x,basis,charge=0)
    mf=scf.RHF(mol)
    ESCF=mf.kernel(verbose=0)
    rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_mo_coeff=rhf_mo_ref,mix_states=mix_states)



    #rhf_mo=mf.mo_coeff
    mf.mo_coeff=rhf_mo
    overlap_basis=block_diag(mol.intor("int1e_ovlp"),mol.intor("int1e_ovlp"))
    energy_basis_1e=block_diag(mol.intor("int1e_kin")+mol.intor("int1e_nuc"),mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
    gmf=scf.addons.convert_to_ghf(mf)
    mo_coeff=gmf.mo_coeff
    fs=mo_coeff.T@gmf.get_fock()@mo_coeff #Fock matrix
    energy_basis_2e=mol.intor('int2e')
    energy_basis_2e_mol_chem=ao2mo.get_mo_eri(energy_basis_2e,(rhf_mo,rhf_mo,rhf_mo,rhf_mo)) #Molecular orbitals in spatial basis, not spin basis. Chemists notation
    spinints_AO_kjemi=make_spinints_aokjemi(dim,energy_basis_2e_mol_chem,alternating)
    spinints_AO_fysikk=np.transpose(spinints_AO_kjemi,(0,2,1,3))
    spinints_AO_fysikk_antisymm=spinints_AO_fysikk-np.transpose(spinints_AO_fysikk,(0,1,3,2))
    spinints=spinints_AO_fysikk_antisymm

    mycc = cc.CCSD(gmf)
    mycc.kernel()
    sample_energies.append(mycc.e_tot)
    t1_pyscf=mycc.t1
    t2_pyscf=mycc.t2
    l1_pyscf,l2_pyscf=mycc.solve_lambda()
    t1s.append(t1_pyscf.T)
    t2s.append(t2_pyscf.T)
    l1s.append(l1_pyscf.T)
    l2s.append(l2_pyscf.T)
    #print("CC pyscf,",mycc.e_corr+ESCF)
    #print("CC own,",ccsdenergy(fs,spinints,t1_pyscf.T,t2_pyscf.T,Nelec,dim)+ESCF)#+ESCF)

#x_alphas=np.linspace(1.5,5,20)
x_alphas=np.linspace(0,4,41)

E_CCSD=[]
E_FCI=[]
E_approx=[]
E_ownmethod=[]
E_diffguess=[]
E_RHF=[]

print("Start EVC")

for x_alpha in x_alphas:
    mol=make_mol(molecule,x_alpha,basis,charge=0)
    mf=scf.RHF(mol)
    ESCF=mf.kernel(verbose=0)
    E_RHF.append(ESCF)
    rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_mo_coeff=rhf_mo_ref,mix_states=mix_states)
    mf_new=scf.RHF(mol)

    #rhf_mo=mf.mo_coeff
    mf.mo_coeff=rhf_mo
    overlap_basis=block_diag(mol.intor("int1e_ovlp"),mol.intor("int1e_ovlp"))
    energy_basis_1e=block_diag(mol.intor("int1e_kin")+mol.intor("int1e_nuc"),mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
    gmf=scf.addons.convert_to_ghf(mf)
    mo_coeff=gmf.mo_coeff
    fs=mo_coeff.T@gmf.get_fock()@mo_coeff #Fock matrix
    energy_basis_2e=mol.intor('int2e')
    energy_basis_2e_mol_chem=ao2mo.get_mo_eri(energy_basis_2e,(rhf_mo,rhf_mo,rhf_mo,rhf_mo)) #Molecular orbitals in spatial basis, not spin basis. Chemists notation
    spinints_AO_kjemi=make_spinints_aokjemi(dim,energy_basis_2e_mol_chem,alternating)
    spinints_AO_fysikk=np.transpose(spinints_AO_kjemi,(0,2,1,3))
    spinints_AO_fysikk_antisymm=spinints_AO_fysikk-np.transpose(spinints_AO_fysikk,(0,1,3,2))
    spinints=spinints_AO_fysikk_antisymm
    #myci = fci.FCI(mol, rhf_mo)
    #e, fcivec = myci.kernel()
    #E_FCI.append(e)
    eb1=np.einsum('pi,pq,qj->ij', rhf_mo, mol.intor("int1e_kin")+mol.intor("int1e_nuc"), rhf_mo)
    ESCF=RHF_energy_nonconverged(Nelec//2,eb1,energy_basis_2e_mol_chem,mol.energy_nuc())
    mycc = cc.CCSD(gmf)
    try:
        mycc.kernel()
        E_CCSD.append(mycc.e_corr+ESCF)
        print(mycc.e_corr)
    except np.linalg.LinAlgError:
        E_CCSD.append(np.nan)
    except:
        E_CCSD.append(np.nan)
    H=np.zeros((len(sample_x),len(sample_x)))
    S=np.zeros((len(sample_x),len(sample_x)))
    Dai = np.zeros((dim,dim))
    for a in range(Nelec,dim):
    	for i in range(0,Nelec):
    		Dai[a,i] = fs[i,i] - fs[a,a]

    # Stanton eq (13)
    Dabij = np.zeros((dim,dim,dim,dim))
    for a in range(Nelec,dim):
    	for b in range(Nelec,dim):
    		for i in range(0,Nelec):
    			for j in range(0,Nelec):
    				Dabij[a,b,i,j] = fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b]

    for i, xi in enumerate(sample_x):
        F,W = updateintermediates(t1s[i],t2s[i],Nelec,dim,fs,spinints)
        t1_error=T1_eq(t1s[i],t2s[i],fs,spinints,F,Nelec,dim,Dai)
        t2_error=T2_eq(t1s[i],t2s[i],fs,spinints,F,Nelec,dim,Dabij,W)
        for j, xj in enumerate(sample_x):
            X1=t1s[i]-t1s[j]
            X2=t2s[i]-t2s[j]
            #overlap=1+np.einsum("ia,ai->",l1s[j].T,X1)+0.25*0.5*np.einsum("ijab,ai,bj->",l2s[j].T,X1,X1)+0.25**2*np.einsum("ijab,abij->",l2s[j].T,X2)
            overlap=1+np.einsum("ia,ai->",l1s[j].T,X1)+0.5*np.einsum("ijab,ai,bj->",l2s[j].T,X1,X1)+0.25*np.einsum("ijab,abij->",l2s[j].T,X2)
            S[i,j]=overlap
            exp_energy=ccsdenergy(fs,spinints,t1s[i],t2s[i],Nelec,dim)+ESCF
            H[i,j]=overlap*exp_energy
            #extra=np.einsum("ia,ai->",l1s[j].T,t1_error)+0.25*np.einsum("ijab,ai,bj->",l2s[j].T,X1,t1_error)+0.25**2*np.einsum("ijab,abij->",l2s[j].T,t2_error)
            extra=np.einsum("ia,ai->",l1s[j].T,t1_error)+np.einsum("ijab,ai,bj->",l2s[j].T,X1,t1_error)+0.25*np.einsum("ijab,abij->",l2s[j].T,t2_error)
            H[i,j]=H[i,j]+extra
    e,cl,c=eig(scipy.linalg.pinv(S,atol=1e-8)@H,left=True)

    idx = np.real(e).argsort()
    e = e[idx]
    c = c[:,idx]
    cl = cl[:,idx]
    E_approx.append(np.real(e[0]))
    t1own,t2own=best_CC_fit(t1s,t2s,np.real(c[:,0])/np.sum(np.real(c[:,0])))
    own_energy=ccsdenergy(fs,spinints,t1own,t2own,Nelec,dim)+ESCF
    #mycc = cc.CCSD(gmf)
    #mycc.kernel(t1=t1own.T,t2=t2own.T)
    #E_diffguess.append(mycc.e_corr+ESCF)
    print("Own:",own_energy,"approx:",np.real(e[0]))#,"difguess:",E_diffguess[-1])
    E_ownmethod.append(own_energy)
print("End EVC")

print("Start approach 2")

energy_simen=[]
start_guess=np.full(len(sample_x),1/len(sample_x))
def orthonormalize_ts(t1s,t2s):
    t_tot=[]
    a,i=t1s[0].shape
    avg_norm=0
    for j in range(len(t1s)):
        t_tot.append(np.concatenate((t1s[j],t2s[j]),axis=None))
        avg_norm+=np.sum(np.abs(t_tot[-1]))
    avg_norm/=len(t1s)
    t_tot=np.array(t_tot).T
    print(t_tot.shape)
    U,s,Vt=svd(t_tot,full_matrices=False)
    t_tot=(U@Vt).T
    """
    for j in range(len(t1s)):
        print(np.sum(np.abs(t_tot[j,:])))
        t_tot[j,:]=t_tot[j,:]/np.sum(np.abs(t_tot[j,:]))*avg_norm
    print(t_tot.shape)
    """
    t1_new=[]
    t2_new=[]
    for j in range(len(t1s)):
        t1_new.append(np.reshape(t_tot[j,:a*i],(a,i)))
        t2_new.append(np.reshape(t_tot[j,a*i:],(a,a,i,i)))
    return t1_new,t2_new
t1s,t2s=orthonormalize_ts(t1s,t2s)
for x_alpha in x_alphas:
    mol=make_mol(molecule,x_alpha,basis,charge=0)
    mf=scf.RHF(mol)
    ESCF=mf.kernel(verbose=0)
    rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_mo_coeff=rhf_mo_ref,mix_states=mix_states)
    #rhf_mo=mf.mo_coeff
    mf.mo_coeff=rhf_mo
    overlap_basis=block_diag(mol.intor("int1e_ovlp"),mol.intor("int1e_ovlp"))
    energy_basis_1e=block_diag(mol.intor("int1e_kin")+mol.intor("int1e_nuc"),mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
    gmf=scf.addons.convert_to_ghf(mf)
    mo_coeff=gmf.mo_coeff
    fs=mo_coeff.T@gmf.get_fock()@mo_coeff #Fock matrix
    energy_basis_2e=mol.intor('int2e')
    energy_basis_2e_mol_chem=ao2mo.get_mo_eri(energy_basis_2e,(rhf_mo,rhf_mo,rhf_mo,rhf_mo)) #Molecular orbitals in spatial basis, not spin basis. Chemists notation
    spinints_AO_kjemi=make_spinints_aokjemi(dim,energy_basis_2e_mol_chem,alternating)
    spinints_AO_fysikk=np.transpose(spinints_AO_kjemi,(0,2,1,3))
    spinints_AO_fysikk_antisymm=spinints_AO_fysikk-np.transpose(spinints_AO_fysikk,(0,1,3,2))
    spinints=spinints_AO_fysikk_antisymm
    eb1=np.einsum('pi,pq,qj->ij', rhf_mo, mol.intor("int1e_kin")+mol.intor("int1e_nuc"), rhf_mo)
    ESCF=RHF_energy_nonconverged(Nelec//2,eb1,energy_basis_2e_mol_chem,mol.energy_nuc())
    Dai = np.zeros((dim,dim))
    for a in range(Nelec,dim):
    	for i in range(0,Nelec):
    		Dai[a,i] = fs[i,i] - fs[a,a]

    # Stanton eq (13)
    Dabij = np.zeros((dim,dim,dim,dim)) #dabei oder "Die bitch? Hmm. Fragen ueber Fragen.
    for a in range(Nelec,dim):
    	for b in range(Nelec,dim):
    		for i in range(0,Nelec):
    			for j in range(0,Nelec):
    				Dabij[a,b,i,j] = fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b]
    #Now the relevant stuff is set up.

    def error_function(params,t1s,t2s,Nelec,dim,fs,spinints,Dai,Dabij,jac=False):
        t1=np.zeros(t1s[0].shape)
        t2=np.zeros(t2s[0].shape)
        for i in range(len(t1s)):
            t1+=params[i]*t1s[i] #Starting guess
            t2+=params[i]*t2s[i] #Starting guess
        projection_errors=np.zeros(len(t1s))

        F,W = updateintermediates(t1,t2,Nelec,dim,fs,spinints)
        t1_error=T1_eq(t1,t2,fs,spinints,F,Nelec,dim,Dai)
        t2_error=T2_eq(t1,t2,fs,spinints,F,Nelec,dim,Dabij,W)
        for i in range(len(projection_errors)):
            projection_errors[i]=np.einsum("ia,ia->",t1_error,t1s[i],optimize=True)+np.einsum("ijab,ijab->",t2_error,t2s[i],optimize=True)
        if (jac==True):
            jacobian=np.zeros((len(params),len(params)))
            delta=1e-14
            for i in range(len(projection_errors)):
                t1_mod=np.zeros(t1s[0].shape)
                t2_mod=np.zeros(t2s[0].shape)
                for k in range(len(t1s)):
                    t1_mod+=params[k]*t1s[k] #Starting guess
                    t2_mod+=params[k]*t2s[k] #Starting guess
                t1_mod+=t1s[i]*delta #Derivative and stuff.
                t2_mod+=t2s[i]*delta #Derivative and stuff.
                Fmod,Wmod = updateintermediates(t1_mod,t2_mod,Nelec,dim,fs,spinints)
                t1_errormod=T1_eq(t1_mod,t2_mod,fs,spinints,Fmod,Nelec,dim,Dai)
                t2_errormod=T2_eq(t1_mod,t2_mod,fs,spinints,Fmod,Nelec,dim,Dabij,Wmod)
                for j in range(len(projection_errors)):
                    jacobian[i,j]=((np.einsum("ia,ia->",t1_errormod,t1s[i],optimize=True)+np.einsum("ijab,ijab->",t2_errormod,t2s[i],optimize=True))-projection_errors[i])/delta
            return projection_errors,jacobian
        return projection_errors
    jacob=False
    sol=root(error_function,start_guess,args=(t1s,t2s,Nelec,dim,fs,spinints,Dai,Dabij,jacob),jac=jacob,method="hybr",options={"xtol":1e-3})#,method="broyden1")
    final=sol.x
    start_guess=final
    print(sol.keys())
    try:
        print("Converged: ",sol.success, " number of iterations:",sol.nit)
    except:
        print("Converged: ",sol.success, " number of iterations:",sol.nfev)
    print(final,sum(final))

    t1=np.zeros(t1s[0].shape)
    t2=np.zeros(t2s[0].shape)
    for i in range(len(t1s)):
        t1+=final[i]*t1s[i] #Starting guess
        t2+=final[i]*t2s[i] #Starting guess
    newEn=ccsdenergy(fs,spinints,t1,t2,Nelec,dim)+ESCF
    energy_simen.append(newEn)
    print(newEn)
plt.plot(x_alphas,E_RHF,label="RHF")
plt.plot(x_alphas,E_CCSD,label="CCSD")
plt.plot(x_alphas,E_approx,label="EVC (approach 1)")
plt.plot(x_alphas,E_ownmethod,label="Fit to approach 1")
plt.plot(x_alphas,energy_simen,label="EVC (approach 2)")
plt.plot(sample_x,sample_energies,"*",label="Sampling points")
plt.xlabel("Distance (Bohr)")
plt.ylabel("Energy (Hartree)")
plt.legend()
plt.tight_layout()
#plt.savefig("%s_%s_%d.pdf"%(molecule_name,basis,len(sample_x)))
plt.show()
