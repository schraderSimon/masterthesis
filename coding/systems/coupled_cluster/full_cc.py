import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo
import sys
np.set_printoptions(linewidth=300,precision=2,suppress=True)
from scipy.linalg import block_diag
from numba import jit
from matrix_operations import *
from helper_functions import *
def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo.T
    B=mo_new.T
    M=B@A.T
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0
def localize_procrustes(mol,mo_coeff,mo_occ,previous_mo_coeff=None):
    mo=mo_coeff[:,mo_occ>0]
    premo=previous_mo_coeff[:,mo_occ>0]
    R,scale=orthogonal_procrustes(mo,premo)
    mo=mo@R

    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo_unocc=mo_coeff[:,mo_occ<=0]
    premo=previous_mo_coeff[:,mo_occ<=0]
    R,scale=orthogonal_procrustes(mo_unocc,premo)
    mo_unocc=mo_unocc@R

    mo_coeff[:,mo_occ<=0]=np.array(mo_unocc)
    #print(mo_unocc)

    return mo_coeff

def make_mol(molecule,x,basis="6-31G",charge=0):
	mol=gto.Mole()
	mol.atom=molecule(x)
	mol.basis = basis
	mol.unit= "Bohr"
	mol.charge=charge
	mol.build()
	return mol
molecule=lambda x: "F 0 0 0; H 0 0 %f"%x
def taus(a,b,i,j,ts,td):
	tausv = td[a,b,i,j] + 0.5*(ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j])
	return tausv

# Stanton eq (10)
def tau(a,b,i,j,ts,td):
	tauv = td[a,b,i,j] + ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j]
	return tauv
def tau_mat(ts,td):
	tau=np.zeros_like(td)
	tau+=td
	tau+=np.einsum("ai,bj->abij",ts,ts)
	tau-=np.einsum("bi,aj->abij",ts,ts)
	return tau
def taus_mat(ts,td):
	tausm=np.zeros_like(td)
	tausm+=td
	tausm+=np.einsum("ai,bj->abij",ts,ts)*0.5
	tausm-=np.einsum("bi,aj->abij",ts,ts)*0.5
	return tausm
def updateintermediates(ts,td,Nelec,dim,fs,spinints,x=True):
	# Stanton eq (3)
	taus_temp=taus_mat(ts,td)
	F = np.zeros((dim,dim))
	F[Nelec:dim,Nelec:dim]=fs[Nelec:dim,Nelec:dim]
	np.fill_diagonal(F[Nelec:dim,Nelec:dim],0)
	W = np.zeros((dim,dim,dim,dim))
	F[Nelec:dim,Nelec:dim]-=0.5*np.einsum("me,am->ae",fs[:Nelec,Nelec:dim],ts[Nelec:dim,:Nelec],optimize=True)
	F[Nelec:dim,Nelec:dim]+=np.einsum("fm,mafe->ae",ts[Nelec:dim,:Nelec],spinints[:Nelec,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	F[Nelec:dim,Nelec:dim]-=0.5*np.einsum("afmn,mnef->ae",taus_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	# Stanton eq (4)
	F[:Nelec,:Nelec]=fs[:Nelec,:Nelec]
	np.fill_diagonal(F[:Nelec,:Nelec],0)
	F[:Nelec,:Nelec]+=0.5*np.einsum("ei,me->mi",ts[Nelec:dim,:Nelec],fs[:Nelec,Nelec:dim],optimize=True)
	F[:Nelec,:Nelec]+=np.einsum("en,mnie->mi",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,:Nelec,Nelec:dim],optimize=True)
	F[:Nelec,:Nelec]+=0.5*np.einsum("efin,mnef->mi",taus_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (5)
	F[:Nelec,Nelec:dim]=fs[:Nelec,Nelec:dim]
	F[:Nelec,Nelec:dim]+=np.einsum("fn,mnef->me",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (6)
	W[:Nelec,:Nelec,:Nelec,:Nelec]=spinints[:Nelec,:Nelec,:Nelec,:Nelec]
	W[:Nelec,:Nelec,:Nelec,:Nelec]+=np.einsum("ej,mnie->mnij",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,:Nelec,Nelec:dim],optimize=True)
	W[:Nelec,:Nelec,:Nelec,:Nelec]-=np.einsum("ei,mnje->mnij",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,:Nelec,Nelec:dim],optimize=True)
	tau_temp=tau_mat(ts,td)
	W[:Nelec,:Nelec,:Nelec,:Nelec]+=0.25*np.einsum("efij,mnef->mnij",tau_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (7)
	W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim] = spinints[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]
	W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]+=np.einsum("am,bmef->abef",ts[Nelec:dim,:Nelec],spinints[Nelec:dim,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]-=np.einsum("bm,amef->abef",ts[Nelec:dim,:Nelec],spinints[Nelec:dim,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]+=0.25*np.einsum("abmn,mnef->abef",tau_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (8)
	Wmbej = np.zeros((dim,dim,dim,dim))
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec] = spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec]
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec]+=np.einsum("fj,mbef->mbej",ts[Nelec:dim,:Nelec],spinints[:Nelec,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec]-=np.einsum("bn,mnej->mbej",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec:,Nelec:dim,:Nelec],optimize=True)
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec]-=0.5*np.einsum("fbjn,mnef->mbej",td[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	W[:Nelec,Nelec:dim,Nelec:dim,:Nelec]-=np.einsum("fj,bn,mnef->mbej",ts[Nelec:dim,:Nelec],ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	return F,W
def T1_eq(ts,td,fs,spinints,F,Nelec,dim,Dai):
	tsnew = np.zeros((dim,dim))
	ts_rel=ts[Nelec:dim,:Nelec]
	tsnew_rel=np.einsum("ai->ia",fs[:Nelec,Nelec:dim],optimize=True)
	tsnew_rel=tsnew_rel+np.einsum("ei,ae->ai",ts_rel,F[Nelec:dim,Nelec:dim],optimize=True)
	tsnew_rel=tsnew_rel-np.einsum("am,mi->ai",ts_rel,F[:Nelec,:Nelec],optimize=True)
	tsnew_rel=tsnew_rel+np.einsum("aeim,me->ai",td[Nelec:dim,Nelec:dim,:Nelec,:Nelec],F[:Nelec,Nelec:dim],optimize=True)
	tsnew_rel=tsnew_rel-0.5*np.einsum("efim,maef->ai",td[Nelec:,Nelec:,:Nelec,:Nelec],spinints[:Nelec,Nelec:,Nelec:,Nelec:],optimize=True)
	tsnew_rel=tsnew_rel-0.5*np.einsum("aemn,nmei->ai",td[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,:Nelec],optimize=True)
	tsnew_rel=tsnew_rel-np.einsum("fn,naif->ai",ts_rel,spinints[:Nelec,Nelec:dim,:Nelec,Nelec:dim],optimize=True)
	tsnew_rel=tsnew_rel-np.einsum("ai,ai->ai",ts[Nelec:dim,:Nelec],Dai[Nelec:dim,:Nelec],optimize=True)
	tsnew[Nelec:dim,:Nelec]=tsnew_rel

	return tsnew

def T2_eq(ts,td,fs,spinints,F,Nelec,dim,Dabij,W):
	tdnew = np.zeros((dim,dim,dim,dim))
	tau_temp=tau_mat(ts,td)
	ts_rel=ts[Nelec:dim,:Nelec]
	td_rel=td[Nelec:dim,Nelec:dim,:Nelec,:Nelec]
	tdnew_rel=np.zeros((dim-Nelec,dim-Nelec,Nelec,Nelec))
	tdnew_rel=np.einsum("abij->ijab",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("aeij,be->abij",td_rel,F[Nelec:dim,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("beij,ae->abij",td_rel,F[Nelec:dim,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-0.5*np.einsum("aeij,bm,me->abij",td_rel,ts_rel,F[:Nelec,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel+0.5*np.einsum("beij,am,me->abij",td_rel,ts_rel,F[:Nelec,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-0.5*np.einsum("abim,ej,me->abij",td_rel,ts_rel,F[:Nelec,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel+0.5*np.einsum("abjm,ei,me->abij",td_rel,ts_rel,F[:Nelec,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("abim,mj->abij",td_rel,F[:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("abjm,mi->abij",td_rel,F[:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("ei,abej->abij",ts_rel,spinints[Nelec:dim,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("ej,abei->abij",ts_rel,spinints[Nelec:dim,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+0.5*np.einsum("efij,abef->abij",tau_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("am,mbij->abij",ts_rel,spinints[:Nelec,Nelec:dim,:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("bm,maij->abij",ts_rel,spinints[:Nelec,Nelec:dim,:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("aeim,mbej->abij",td_rel,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("ei,am,mbej->abij",ts_rel,ts_rel,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("aejm,mbei->abij",td_rel,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("ej,am,mbei->abij",ts_rel,ts_rel,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("beim,maej->abij",td_rel,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("ei,bm,maej->abij",ts_rel,ts_rel,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("bejm,maei->abij",td_rel,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("ej,bm,maei->abij",ts_rel,ts_rel,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+0.5*np.einsum("abmn,mnij->abij",tau_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],W[:Nelec,:Nelec,:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("abij,abij->abij",td_rel,Dabij[Nelec:dim,Nelec:dim,:Nelec,:Nelec],optimize=True)
	tdnew[Nelec:dim,Nelec:dim,:Nelec,:Nelec]=tdnew_rel
	return tdnew
def makeT1(ts,td,fs,spinints,F,Nelec,dim,Dai,x=True):

	tsnew = np.zeros((dim,dim))
	tsnew_rel=np.zeros((dim-Nelec,Nelec))
	ts_rel=ts[Nelec:dim,:Nelec]
	tsnew_rel=np.einsum("ai->ia",fs[:Nelec,Nelec:dim],optimize=True)
	tsnew_rel=tsnew_rel+np.einsum("ei,ae->ai",ts_rel,F[Nelec:dim,Nelec:dim],optimize=True)
	tsnew_rel=tsnew_rel-np.einsum("am,mi->ai",ts_rel,F[:Nelec,:Nelec],optimize=True)
	tsnew_rel=tsnew_rel+np.einsum("aeim,me->ai",td[Nelec:dim,Nelec:dim,:Nelec,:Nelec],F[:Nelec,Nelec:dim],optimize=True)
	tsnew_rel=tsnew_rel-0.5*np.einsum("efim,maef->ai",td[Nelec:,Nelec:,:Nelec,:Nelec],spinints[:Nelec,Nelec:,Nelec:,Nelec:],optimize=True)
	tsnew_rel=tsnew_rel-0.5*np.einsum("aemn,nmei->ai",td[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,:Nelec],optimize=True)
	tsnew_rel=tsnew_rel-np.einsum("fn,naif->ai",ts_rel,spinints[:Nelec,Nelec:dim,:Nelec,Nelec:dim],optimize=True)
	tsnew_rel=tsnew_rel/Dai[Nelec:dim,:Nelec]
	tsnew[Nelec:dim,:Nelec]=tsnew_rel
	return tsnew
# Stanton eq (2)
def makeT2(ts,td,fs,spinints,F,Nelec,dim,Dabij,W):
	tau_temp=tau_mat(ts,td)
	tdnew = np.zeros((dim,dim,dim,dim))
	ts_rel=ts[Nelec:dim,:Nelec]
	td_rel=td[Nelec:dim,Nelec:dim,:Nelec,:Nelec]
	tdnew_rel=np.zeros((dim-Nelec,dim-Nelec,Nelec,Nelec))
	tdnew_rel=np.einsum("abij->ijab",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("aeij,be->abij",td_rel,F[Nelec:dim,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("beij,ae->abij",td_rel,F[Nelec:dim,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-0.5*np.einsum("aeij,bm,me->abij",td_rel,ts_rel,F[:Nelec,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel+0.5*np.einsum("beij,am,me->abij",td_rel,ts_rel,F[:Nelec,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-0.5*np.einsum("abim,ej,me->abij",td_rel,ts_rel,F[:Nelec,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel+0.5*np.einsum("abjm,ei,me->abij",td_rel,ts_rel,F[:Nelec,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("abim,mj->abij",td_rel,F[:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("abjm,mi->abij",td_rel,F[:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("ei,abej->abij",ts_rel,spinints[Nelec:dim,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("ej,abei->abij",ts_rel,spinints[Nelec:dim,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+0.5*np.einsum("efij,abef->abij",tau_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],W[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("am,mbij->abij",ts_rel,spinints[:Nelec,Nelec:dim,:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("bm,maij->abij",ts_rel,spinints[:Nelec,Nelec:dim,:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("aeim,mbej->abij",td_rel,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("ei,am,mbej->abij",ts_rel,ts_rel,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("aejm,mbei->abij",td_rel,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("ej,am,mbei->abij",ts_rel,ts_rel,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("beim,maej->abij",td_rel,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("ei,bm,maej->abij",ts_rel,ts_rel,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+np.einsum("bejm,maei->abij",td_rel,W[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel-np.einsum("ej,bm,maei->abij",ts_rel,ts_rel,spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel+0.5*np.einsum("abmn,mnij->abij",tau_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],W[:Nelec,:Nelec,:Nelec,:Nelec],optimize=True)
	tdnew_rel=tdnew_rel/Dabij[Nelec:dim,Nelec:dim,:Nelec,:Nelec]
	tdnew[Nelec:dim,Nelec:dim,:Nelec,:Nelec]=tdnew_rel
	return tdnew
# Expression from Crawford, Schaefer (2000)
# DOI: 10.1002/9780470125915.ch2
# Equation (134) and (173)
# computes CCSD energy given T1 and T2

def ccsdenergy(fs,spinints,ts,td,Nelec,dim):
	ECCSD = np.einsum("ia,ai->",fs[:Nelec,Nelec:dim],ts[Nelec:dim,:Nelec],optimize=True)
	ECCSD+=0.25*np.einsum("ijab,abij->",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],td[Nelec:dim,Nelec:dim,:Nelec,:Nelec],optimize=True)
	ECCSD+=0.5*np.einsum("ijab,ai,bj->",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],ts[Nelec:dim,:Nelec],ts[Nelec:dim,:Nelec],optimize=True)
	return ECCSD



#Step 1: Produce Fock matrix

basis="6-31G"
molecule_name="HF"
x_sol=np.linspace(1.5,4,30)
ref_x=2/1.1386276671
#x_sol=np.array([1.9,2.5])
energies_ref=np.zeros(len(x_sol))
mol=make_mol(molecule,ref_x,basis,charge=0)
ENUC=mol.energy_nuc()
Nelec=mol.nelectron

print(ENUC)
mf=scf.RHF(mol)
ESCF=mf.kernel()
rhf_mo_ref=mf.mo_coeff

xnew=2*ref_x
mol=make_mol(molecule,xnew,basis,charge=0)
mf=scf.RHF(mol)
ESCF=mf.kernel()
rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,previous_mo_coeff=rhf_mo_ref)
#rhf_mo=mf.mo_coeff


mf.mo_coeff=rhf_mo
overlap_basis=block_diag(mol.intor("int1e_ovlp"),mol.intor("int1e_ovlp"))
energy_basis_1e=block_diag(mol.intor("int1e_kin")+mol.intor("int1e_nuc"),mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
gmf=scf.addons.convert_to_ghf(mf)


energy_basis_2e=mol.intor('int2e')

mycc = cc.CCSD(gmf)
mycc.kernel()
t1_pyscf=mycc.t1

mo_coeff=gmf.mo_coeff
fs=mo_coeff.T@gmf.get_fock()@mo_coeff #Fock matrix
print(fs)
energy_basis_1eMO = np.einsum('pi,pq,qj->ij', mo_coeff, energy_basis_1e, mo_coeff)
dim=len(energy_basis_2e)*2
dimhalf=len(energy_basis_2e)
energy_basis_2e_mol_chem=ao2mo.get_mo_eri(energy_basis_2e,(rhf_mo,rhf_mo,rhf_mo,rhf_mo)) #Molecular orbitals in spatial basis, not spin basis. Chemists notation
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
				spinints_AO_kjemi[p,q,r,s]=energy_basis_2e_mol_chem[P,Q,R,S]*(spinP==spinQ)*(spinR==spinS)
spinints_AO_fysikk=np.transpose(spinints_AO_kjemi,(0,2,1,3))
spinints_AO_fysikk_antisymm=spinints_AO_fysikk-np.transpose(spinints_AO_fysikk,(0,1,3,2))
spinints=spinints_AO_fysikk_antisymm

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

ts = np.zeros((dim,dim))
td = np.zeros((dim,dim,dim,dim))

# Initial guess T2 --- from MP2 calculation!

for a in range(Nelec,dim):
	for b in range(Nelec,dim):
		for i in range(0,Nelec):
			for j in range(0,Nelec):
				td[a,b,i,j] += spinints[i,j,a,b]/(fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b])

ECCSD = 0
DECC = 1
counter=0
while DECC > 1e-14: # arbitrary convergence criteria
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
print(ts[Nelec:dim,:Nelec].T)
print(mycc.t1)
print("Difference in T1 coefficients: %e"%(np.max(np.abs(ts[Nelec:dim,:Nelec].T-mycc.t1))))
print("Difference in T2 coefficients: %e"%(np.max(np.abs(td[Nelec:dim,Nelec:dim,:Nelec,:Nelec].T-mycc.t2))))
