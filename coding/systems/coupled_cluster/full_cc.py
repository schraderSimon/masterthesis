import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo,fci
import sys
np.set_printoptions(linewidth=300,precision=4,suppress=True)
from scipy.linalg import block_diag, eig
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



#Step 1: Produce Fock matrix

basis="6-31G"
molecule_name="HF"
x_sol=np.linspace(1.5,4,30)
ref_x=2
#x_sol=np.array([1.9,2.5])
energies_ref=np.zeros(len(x_sol))
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
rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,previous_mo_coeff=rhf_mo_ref)
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
sample_x=[1.0,1.4,1.7,2.0,2.3,2.4,2.5]
t1s=[]
t2s=[]
l1s=[]
l2s=[]
for x in sample_x:
    mol=make_mol(molecule,x,basis,charge=0)
    mf=scf.RHF(mol)
    ESCF=mf.kernel(verbose=0)
    rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,previous_mo_coeff=rhf_mo_ref)
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
    t1_pyscf=mycc.t1
    t2_pyscf=mycc.t2
    t1s.append(t1_pyscf.T)
    t2s.append(t2_pyscf.T)
    l1s.append(l1_pyscf.T)
    l2s.append(l2_pyscf.T)
    #print("CC pyscf,",mycc.e_corr+ESCF)
    #print("CC own,",ccsdenergy(fs,spinints,t1_pyscf.T,t2_pyscf.T,Nelec,dim)+ESCF)#+ESCF)

    l1_pyscf,l2_pyscf=mycc.solve_lambda()
x_alphas=np.linspace(1.5,5.0,36)
E_CCSD=[]
E_FCI=[]
E_approx=[]
print("Start EVC")
for x_alpha in x_alphas:
    mol=make_mol(molecule,x_alpha,basis,charge=0)
    mf=scf.RHF(mol)
    ESCF=mf.kernel(verbose=0)
    rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,previous_mo_coeff=rhf_mo_ref)
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
    myci = fci.FCI(mol, rhf_mo)
    e, fcivec = myci.kernel()
    E_FCI.append(e)
    mycc = cc.CCSD(gmf)
    try:
        mycc.kernel()
        E_CCSD.append(mycc.e_corr+ESCF)
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
            overlap=1+np.einsum("ia,ai->",l1s[j].T,X1)+0.5*np.einsum("ijab,ai,bj->",l2s[j].T,X1,X1)+0.25*np.einsum("ijab,abij->",l2s[j].T,X2)
            S[i,j]=overlap
            exp_energy=ccsdenergy(fs,spinints,t1s[i],t2s[i],Nelec,dim)+ESCF
            H[i,j]=overlap*exp_energy
            extra=np.einsum("ia,ai->",l1s[j].T,t1_error)+np.einsum("ijab,ai,bj->",l2s[j].T,X1,t1_error)+0.25*np.einsum("ijab,abij->",l2s[j].T,t2_error)
            H[i,j]=H[i,j]+extra
    e,c=eig(H,b=S)
    E_approx.append(np.min(np.real(e)))
print("End EVC")
plt.plot(x_alphas,E_CCSD,label="CCSD")
plt.plot(x_alphas,E_approx,label="EVC")
plt.plot(x_alphas,E_FCI,label="FCI")
plt.legend()
plt.show()
