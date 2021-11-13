import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo
import sys
np.set_printoptions(linewidth=300,precision=2,suppress=True)
from scipy.linalg import block_diag
from numba import jit
def make_mol(molecule,x,basis="6-31G",charge=0):
	mol=gto.Mole()
	mol.atom=molecule(x)
	mol.basis = basis
	mol.unit= "Bohr"
	mol.charge=charge
	mol.build()
	return mol
molecule=lambda x: "H 0 0 0; F 0 0 %f"%x

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
	Fae = np.zeros((dim,dim))
	Fae[Nelec:dim,Nelec:dim]=fs[Nelec:dim,Nelec:dim]
	np.fill_diagonal(Fae,0)
	Fae[Nelec:dim,Nelec:dim]-=0.5*np.einsum("me,am->ae",fs[:Nelec,Nelec:dim],ts[Nelec:dim,:Nelec],optimize=True)
	Fae[Nelec:dim,Nelec:dim]+=np.einsum("fm,mafe->ae",ts[Nelec:dim,:Nelec],spinints[:Nelec,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	Fae[Nelec:dim,Nelec:dim]-=0.5*np.einsum("afmn,mnef->ae",taus_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	# Stanton eq (4)
	Fmi = np.zeros((dim,dim))
	Fmi[:Nelec,:Nelec]=fs[:Nelec,:Nelec]
	np.fill_diagonal(Fmi,0)
	Fmi[:Nelec,:Nelec]+=0.5*np.einsum("ei,me->mi",ts[Nelec:dim,:Nelec],fs[:Nelec,Nelec:dim],optimize=True)
	Fmi[:Nelec,:Nelec]+=np.einsum("en,mnie->mi",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,:Nelec,Nelec:dim],optimize=True)
	Fmi[:Nelec,:Nelec]+=0.5*np.einsum("efin,mnef->mi",taus_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (5)
	Fme = np.zeros((dim,dim))
	Fme[:Nelec,Nelec:dim]=fs[:Nelec,Nelec:dim]
	Fme[:Nelec,Nelec:dim]+=np.einsum("fn,mnef->me",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (6)
	Wmnij = np.zeros((dim,dim,dim,dim))
	Wmnij[:Nelec,:Nelec,:Nelec,:Nelec]=spinints[:Nelec,:Nelec,:Nelec,:Nelec]
	Wmnij[:Nelec,:Nelec,:Nelec,:Nelec]+=np.einsum("ej,mnie->mnij",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,:Nelec,Nelec:dim],optimize=True)
	Wmnij[:Nelec,:Nelec,:Nelec,:Nelec]-=np.einsum("ei,mnje->mnij",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,:Nelec,Nelec:dim],optimize=True)
	tau_temp=tau_mat(ts,td)
	Wmnij[:Nelec,:Nelec,:Nelec,:Nelec]+=0.25*np.einsum("efij,mnef->mnij",tau_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (7)
	Wabef = np.zeros((dim,dim,dim,dim))
	Wabef[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim] = spinints[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]
	Wabef[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]+=np.einsum("am,bmef->abef",ts[Nelec:dim,:Nelec],spinints[Nelec:dim,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	Wabef[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]-=np.einsum("bm,amef->abef",ts[Nelec:dim,:Nelec],spinints[Nelec:dim,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	Wabef[Nelec:dim,Nelec:dim,Nelec:dim,Nelec:dim]+=0.25*np.einsum("abmn,mnef->abef",tau_temp[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	# Stanton eq (8)
	Wmbej = np.zeros((dim,dim,dim,dim))
	Wmbej[:Nelec,Nelec:dim,Nelec:dim,:Nelec] = spinints[:Nelec,Nelec:dim,Nelec:dim,:Nelec]
	Wmbej[:Nelec,Nelec:dim,Nelec:dim,:Nelec]+=np.einsum("fj,mbef->mbej",ts[Nelec:dim,:Nelec],spinints[:Nelec,Nelec:dim,Nelec:dim,Nelec:dim],optimize=True)
	Wmbej[:Nelec,Nelec:dim,Nelec:dim,:Nelec]-=np.einsum("bn,mnej->mbej",ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec:,Nelec:dim,:Nelec],optimize=True)
	Wmbej[:Nelec,Nelec:dim,Nelec:dim,:Nelec]-=0.5*np.einsum("fbjn,mnef->mbej",td[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)
	Wmbej[:Nelec,Nelec:dim,Nelec:dim,:Nelec]-=np.einsum("fj,bn,mnef->mbej",ts[Nelec:dim,:Nelec],ts[Nelec:dim,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],optimize=True)

	return Fae, Fmi, Fme, Wmnij, Wabef, Wmbej
def T1_eq(ts,td,fs,spinints,Fae,Fmi,Fme,Nelec,dim,Dai):
	tsnew = np.zeros((dim,dim))
	for a in range(Nelec,dim):
		for i in range(0,Nelec):
			tsnew[a,i] = fs[i,a]
			for e in range(Nelec,dim):
	 			tsnew[a,i] += ts[e,i]*Fae[a,e]
			for m in range(0,Nelec):
				tsnew[a,i] += -ts[a,m]*Fmi[m,i]
				for e in range(Nelec,dim):
					tsnew[a,i] += td[a,e,i,m]*Fme[m,e]
					for f in range(Nelec,dim):
	 					tsnew[a,i] += -0.5*td[e,f,i,m]*spinints[m,a,e,f]
					for n in range(0,Nelec):
						tsnew[a,i] += -0.5*td[a,e,m,n]*spinints[n,m,e,i]
			for n in range(0,Nelec):
				for f in range(Nelec,dim):
					tsnew[a,i] += -ts[f,n]*spinints[n,a,i,f]
			tsnew[a,i] -= ts[a,i]*Dai[a,i]
	return tsnew

def T2_eq(ts,td,fs,spinints,Fae,Fmi,Fme,Wmnij,Wabef,Wmbej,Nelec,dim,Dabij):
	val = np.zeros((dim,dim,dim,dim))
	for a in range(Nelec,dim):
		for b in range(Nelec,dim):
			for i in range(0,Nelec):
				for j in range(0,Nelec):
					val[a,b,i,j] += spinints[i,j,a,b]
					for e in range(Nelec,dim):
						val[a,b,i,j] += td[a,e,i,j]*Fae[b,e] - td[b,e,i,j]*Fae[a,e]
						for m in range(0,Nelec):
							val[a,b,i,j] += -0.5*td[a,e,i,j]*ts[b,m]*Fme[m,e] + 0.5*td[b,e,i,j]*ts[a,m]*Fme[m,e]
							continue
					for m in range(0,Nelec):
						val[a,b,i,j] += -td[a,b,i,m]*Fmi[m,j] + td[a,b,j,m]*Fmi[m,i]
						for e in range(Nelec,dim):
							val[a,b,i,j] += -0.5*td[a,b,i,m]*ts[e,j]*Fme[m,e] + 0.5*td[a,b,j,m]*ts[e,i]*Fme[m,e]
							continue
					for e in range(Nelec,dim):
						val[a,b,i,j] += ts[e,i]*spinints[a,b,e,j] - ts[e,j]*spinints[a,b,e,i]
						for f in range(Nelec,dim):
							val[a,b,i,j] += 0.5*tau(e,f,i,j,ts,td)*Wabef[a,b,e,f]
							continue
					for m in range(0,Nelec):
						val[a,b,i,j] += -ts[a,m]*spinints[m,b,i,j] + ts[b,m]*spinints[m,a,i,j]
						for e in range(Nelec,dim):
							val[a,b,i,j] +=  td[a,e,i,m]*Wmbej[m,b,e,j] - ts[e,i]*ts[a,m]*spinints[m,b,e,j]
							val[a,b,i,j] += -td[a,e,j,m]*Wmbej[m,b,e,i] + ts[e,j]*ts[a,m]*spinints[m,b,e,i]
							val[a,b,i,j] += -td[b,e,i,m]*Wmbej[m,a,e,j] + ts[e,i]*ts[b,m]*spinints[m,a,e,j]
							val[a,b,i,j] +=  td[b,e,j,m]*Wmbej[m,a,e,i] - ts[e,j]*ts[b,m]*spinints[m,a,e,i]
							continue
						for n in range(0,Nelec):
							val[a,b,i,j] += 0.5*tau(a,b,m,n,ts,td)*Wmnij[m,n,i,j]
							continue
					val[a,b,i,j] -= td[a,b,i,j]*Dabij[a,b,i,j]
	return val
def makeT1(ts,td,fs,spinints,Fae,Fmi,Fme,Nelec,dim,Dai,x=True):
	tsnew = np.zeros((dim,dim))
	tsnew[Nelec:dim,:Nelec]=np.einsum("ai->ia",fs[:Nelec,Nelec:dim],optimize=True)
	tsnew[Nelec:dim,:Nelec]+=np.einsum("ei,ae->ai",ts[Nelec:dim,:Nelec],Fae[Nelec:dim,Nelec:dim],optimize=True)
	tsnew[Nelec:dim,:Nelec]-=np.einsum("am,mi->ai",ts[Nelec:dim,:Nelec],Fmi[:Nelec,:Nelec],optimize=True)
	tsnew[Nelec:dim,:Nelec]+=np.einsum("aeim,me->ai",td[Nelec:dim,Nelec:dim,:Nelec,:Nelec],Fme[:Nelec,Nelec:dim],optimize=True)
	tsnew[Nelec:dim,:Nelec]-=0.5*np.einsum("efim,maef->ai",td[Nelec:,Nelec:,:Nelec,:Nelec],spinints[:Nelec,Nelec:,Nelec:,Nelec:],optimize=True)
	tsnew[Nelec:dim,:Nelec]-=0.5*np.einsum("aemn,nmei->ai",td[Nelec:dim,Nelec:dim,:Nelec,:Nelec],spinints[:Nelec,:Nelec,Nelec:dim,:Nelec],optimize=True)
	tsnew[Nelec:dim,:Nelec]-=np.einsum("fn,naif->ai",ts[Nelec:dim,:Nelec],spinints[:Nelec,Nelec:dim,:Nelec,Nelec:dim],optimize=True)
	tsnew/=Dai
	return tsnew
# Stanton eq (2)
def makeT2(ts,td,fs,spinints,Fae,Fmi,Fme,Wmnij,Wabef,Wmbej,Nelec,dim,Dabij,x=True):
	tdnew = np.zeros((dim,dim,dim,dim))
	for a in range(Nelec,dim):
		for b in range(Nelec,dim):
			for i in range(0,Nelec):
				for j in range(0,Nelec):
					tdnew[a,b,i,j] += spinints[i,j,a,b]
					for e in range(Nelec,dim):
						tdnew[a,b,i,j] += td[a,e,i,j]*Fae[b,e] - td[b,e,i,j]*Fae[a,e]
						for m in range(0,Nelec):
							tdnew[a,b,i,j] += -0.5*td[a,e,i,j]*ts[b,m]*Fme[m,e] + 0.5*td[b,e,i,j]*ts[a,m]*Fme[m,e]
							continue
					for m in range(0,Nelec):
						tdnew[a,b,i,j] += -td[a,b,i,m]*Fmi[m,j] + td[a,b,j,m]*Fmi[m,i]
						for e in range(Nelec,dim):
							tdnew[a,b,i,j] += -0.5*td[a,b,i,m]*ts[e,j]*Fme[m,e] + 0.5*td[a,b,j,m]*ts[e,i]*Fme[m,e]
							continue
					for e in range(Nelec,dim):
						tdnew[a,b,i,j] += ts[e,i]*spinints[a,b,e,j] - ts[e,j]*spinints[a,b,e,i]
						for f in range(Nelec,dim):
							tdnew[a,b,i,j] += 0.5*tau(e,f,i,j,ts,td)*Wabef[a,b,e,f]
							continue
					for m in range(0,Nelec):
						tdnew[a,b,i,j] += -ts[a,m]*spinints[m,b,i,j] + ts[b,m]*spinints[m,a,i,j]
						for e in range(Nelec,dim):
							tdnew[a,b,i,j] +=  td[a,e,i,m]*Wmbej[m,b,e,j] - ts[e,i]*ts[a,m]*spinints[m,b,e,j]
							tdnew[a,b,i,j] += -td[a,e,j,m]*Wmbej[m,b,e,i] + ts[e,j]*ts[a,m]*spinints[m,b,e,i]
							tdnew[a,b,i,j] += -td[b,e,i,m]*Wmbej[m,a,e,j] + ts[e,i]*ts[b,m]*spinints[m,a,e,j]
							tdnew[a,b,i,j] +=  td[b,e,j,m]*Wmbej[m,a,e,i] - ts[e,j]*ts[b,m]*spinints[m,a,e,i]
							continue
						for n in range(0,Nelec):
							tdnew[a,b,i,j] += 0.5*tau(a,b,m,n,ts,td)*Wmnij[m,n,i,j]
							continue
					tdnew[a,b,i,j] = tdnew[a,b,i,j]/Dabij[a,b,i,j]
	return tdnew
# Expression from Crawford, Schaefer (2000)
# DOI: 10.1002/9780470125915.ch2
# Equation (134) and (173)
# computes CCSD energy given T1 and T2

def ccsdenergy(fs,spinints,ts,td,Nelec,dim):
	ECCSD = np.einsum("ia,ai->",fs[:Nelec,Nelec:dim],ts[Nelec:dim,:Nelec])
	ECCSD+=0.25*np.einsum("ijab,abij->",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],td[Nelec:dim,Nelec:dim,:Nelec,:Nelec])
	ECCSD+=0.5*np.einsum("ijab,ai,bj->",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],ts[Nelec:dim,:Nelec],ts[Nelec:dim,:Nelec])
	return ECCSD



#Step 1: Produce Fock matrix

basis="STO-3G"
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
rhf_mo=mf.mo_coeff
rhf_mo[:,1]=-rhf_mo[:,1]
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
while DECC > 1e-12 : # arbitrary convergence criteria
    OLDCC = ECCSD
    Fae,Fmi,Fme,Wmnij,Wabef,Wmbej = updateintermediates(ts,td,Nelec,dim,fs,spinints)
    tsnew = makeT1(ts,td,fs,spinints,Fae,Fmi,Fme,Nelec,dim,Dai)
    tdnew = makeT2(ts,td,fs,spinints,Fae,Fmi,Fme,Wmnij,Wabef,Wmbej,Nelec,dim,Dabij)
    ts = tsnew
    td = tdnew
    ECCSD = ccsdenergy(fs,spinints,ts,td,Nelec,dim)
    #print(DECC)
    DECC = abs(ECCSD - OLDCC)
    t1_val=T1_eq(ts,td,fs,spinints,Fae,Fmi,Fme,Nelec,dim,Dai)
    t2_val=T2_eq(ts,td,fs,spinints,Fae,Fmi,Fme,Wmnij,Wabef,Wmbej,Nelec,dim,Dabij)
    print("Max T2-error",np.max(abs(t2_val)))
    print("Max T1-error",np.max(abs(t1_val)))
print("E(corr,CCSD) = ", ECCSD)
print("E(CCSD) = ", ECCSD + ESCF)
#print(spinints)
