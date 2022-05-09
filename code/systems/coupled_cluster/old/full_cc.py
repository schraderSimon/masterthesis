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
from sklearn.decomposition import PCA
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/coupled_cluster")
from rccsd_gs import orthonormalize_ts
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
    t1_new=np.reshape(x_sol[:a*i],(a,i))
    t2_new=np.reshape(x_sol[a*i:],(a,a,i,i))
    return t1_new,t2_new

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

"""
def orthonormalize_ts_choice(t1s,t2s,choice):
    t_tot=[]
    t_tot_small=[]
    a,i=t1s[0].shape
    avg_norm=0
    for j in range(len(t1s)):
        #t1_contr=np.einsum("ai,bk->abik",t1s[j],t1s[j])
        #t_tot.append(np.concatenate((t1s[j],0.5*t1_contr+0.25*t2s[j]),axis=None))
        t_tot_small.append(np.concatenate((t1s[j],t2s[j]),axis=None)[choice])
        t_tot.append(np.concatenate((t1s[j],t2s[j]),axis=None))
        avg_norm+=np.sum(np.abs(t_tot[-1]))
    t_tot_small=np.array(t_tot_small).T
    print(t_tot_small.shape)
    t_tot=np.array(t_tot).T
    M=t_tot_small.T@t_tot_small
    Mpowerminushalf=powerminushalf(M)
    t_tot=(t_tot@Mpowerminushalf).T
    print(t_tot.shape)
    t1_new=[]
    t2_new=[]
    for j in range(len(t1s)):
        new_t1=np.reshape(t_tot[j,:a*i],(a,i))
        #new_t2=(4*np.reshape(t_tot[j,a*i:],(a,a,i,i))-2*np.einsum("ai,bk->abik",new_t1,new_t1))
        new_t2=np.reshape(t_tot[j,a*i:],(a,a,i,i))
        t1_new.append(new_t1)
        t2_new.append(new_t2)
    return t1_new,t2_new
def orthonormalize_ts_pca(t1s,t2s,nos,nvs):

    t_tot=[]
    a,i=t1s[0].shape
    avg_norm=0
    for j in range(len(t1s)):
        #t1_contr=np.einsum("ai,bk->abik",t1s[j],t1s[j])
        #t_tot.append(np.concatenate((t1s[j],0.5*t1_contr+0.25*t2s[j]),axis=None))
        t_tot.append(np.concatenate((t1s[j],t2s[j]),axis=None))
        avg_norm+=np.sum(np.abs(t_tot[-1]))
    avg_norm/=len(t1s)
    t_tot=np.array(t_tot).T
    #t0_old=np.concatenate((t1s[0],t2s[0][np.ix_(nvs,nvs,nos,nos)]),axis=None)
    t0_old=np.concatenate((t1s[0],t2s[0]),axis=None)
    pca=PCA()
    #print(t_tot.shape)
    #print(t_tot.T@t_tot)
    t_tot=np.array(pca.fit_transform(t_tot)).T
    diags=np.sqrt(np.diag(t_tot@t_tot.T))**(-1)
    print(diags)
    t_tot=(diags*t_tot.T).T
    diags=np.sqrt(np.diag(t_tot@t_tot.T))
    #print(t_tot.shape)
    print(diags)
    #U,s,Vt=svd(t_tot,full_matrices=False)
    #t_tot=(U@Vt).T

    t1_new=[]
    t2_new=[]
    for j in range(len(t1s)):
        new_t1=np.reshape(t_tot[j,:a*i],(a,i))
        new_t2=np.reshape(t_tot[j,a*i:],(a,a,i,i))
        t1_new.append(new_t1)
        t2_new.append(new_t2)
    start_guess=[]
    for i in range(len(t1s)):
        #start_guess.append(t0_old@np.concatenate((t1_new[i],t2_new[i][np.ix_(nvs,nvs,nos,nos)]),axis=None))
        start_guess.append(t0_old@np.concatenate((t1_new[i],t2_new[i]),axis=None))
    print(start_guess)
    return t1_new,t2_new, start_guess
"""
def make_mol(molecule,x,basis="6-31G",charge=0):
	mol=gto.Mole()
	mol.atom=molecule(*x)
	mol.basis = basis
	mol.unit= "Bohr"
	mol.charge=charge
	mol.build()
	return mol
#molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
class GCC_solver():
    def __init__(self,mol,reference_state=None,mix_states=False,type="procrustes"):
        self.mol=mol
        self.Nelec=mol.nelectron
        self.basis=mol.basis
        self.fs=None #The Fock matrix of the (solved) molecule
        self.spinints=None #The spin integrals
        self.overlap_basis=block_diag(mol.intor("int1e_ovlp"),mol.intor("int1e_ovlp"))
        self.energy_basis_1e=block_diag(mol.intor("int1e_kin")+mol.intor("int1e_nuc"),mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
        self.energy_basis_2e=mol.intor('int2e')
        self.dim=len(self.energy_basis_1e)
        self.enuc=mol.energy_nuc()
        self.reference_state=reference_state
        self.mix_states=mix_states
        self.type=type
        self.grhfbasis()
    def grhfbasis(self):
        mol=self.mol
        Nelec=self.Nelec
        dim=self.dim
        mf=scf.RHF(mol)
        self.ESCF=mf.kernel()
        if self.reference_state is not None:
            if self.type=="basischange":
                rhf_mo=basischange(self.reference_state,mol.intor("int1e_ovlp"),mol.nelectron//2)
            else:
                rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_mo_coeff=self.reference_state,mix_states=self.mix_states)

        else:
            rhf_mo=mf.mo_coeff
        mf.mo_coeff=rhf_mo
        gmf=scf.addons.convert_to_ghf(mf)
        self.gmf=gmf
        mo_coeff=gmf.mo_coeff
        self.fs=mo_coeff.T@gmf.get_fock()@mo_coeff #Fock matrix
        self.energy_basis_1eMO = np.einsum('pi,pq,qj->ij', rhf_mo, self.mol.intor("int1e_kin")+mol.intor("int1e_nuc"), rhf_mo)
        self.energy_basis_2e_mol_chem=ao2mo.get_mo_eri(self.energy_basis_2e,(rhf_mo,rhf_mo,rhf_mo,rhf_mo)) #Molecular orbitals in spatial basis, not spin basis. Chemists notation
        alternating=np.array([[(i+j)%2 for i in range(1,dim+1)] for j in range(dim)])
        self.spinints_AO_kjemi=make_spinints_aokjemi(dim,self.energy_basis_2e_mol_chem,alternating)
        spinints_AO_fysikk=np.transpose(self.spinints_AO_kjemi,(0,2,1,3))
        spinints_AO_fysikk_antisymm=spinints_AO_fysikk-np.transpose(spinints_AO_fysikk,(0,1,3,2))
        self.spinints=spinints_AO_fysikk_antisymm
        self.mo_coeff=mo_coeff
        Dai = np.zeros((dim,dim))
        for a in range(Nelec,dim):
        	for i in range(0,Nelec):
        		Dai[a,i] = self.fs[i,i] - self.fs[a,a]

        # Stanton eq (13)
        Dabij = np.zeros((dim,dim,dim,dim))
        for a in range(Nelec,dim):
        	for b in range(Nelec,dim):
        		for i in range(0,Nelec):
        			for j in range(0,Nelec):
        				Dabij[a,b,i,j] = self.fs[i,i] + self.fs[j,j] - self.fs[a,a] - self.fs[b,b]
        self.Dai=Dai
        self.Dabij=Dabij
        self.ESCF=self.RHF_energy_nonconverged()
    def rhfbasis(self):
        mol=self.mol
        Nelec=self.Nelec
        dim=self.dim
        mf=scf.RHF(mol)
        self.ESCF=mf.kernel()
        if self.reference_state is not None:
            if self.type=="basischange":
                rhf_mo=basischange(self.reference_state,mol.intor("int1e_ovlp"),mol.nelectron//2)
            else:
                rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_mo_coeff=self.reference_state,mix_states=self.mix_states)
        else:
            rhf_mo=mf.mo_coeff
        mf.mo_coeff=rhf_mo
        mo_coeff=rhf_mo
        self.fs=mo_coeff.T@mf.get_fock()@mo_coeff #Fock matrix
        self.energy_basis_1eMO = np.einsum('pi,pq,qj->ij', rhf_mo, self.mol.intor("int1e_kin")+mol.intor("int1e_nuc"), rhf_mo)
        self.energy_basis_2e_mol_chem=ao2mo.get_mo_eri(self.energy_basis_2e,(rhf_mo,rhf_mo,rhf_mo,rhf_mo)) #Molecular orbitals in spatial basis, not spin basis. Chemists notation
        self.spinints_AO_kjemi=self.energy_basis_2e_mol_chem
        spinints_AO_fysikk=np.transpose(self.spinints_AO_kjemi,(0,2,1,3))
        spinints_AO_fysikk_antisymm=spinints_AO_fysikk-np.transpose(spinints_AO_fysikk,(0,1,3,2))
        self.spinints=spinints_AO_fysikk_antisymm
        self.mo_coeff=mo_coeff

    def RHF_energy_nonconverged(self):
        E=self.enuc
        Nelec2=self.Nelec//2
        for i in range(Nelec2):
            E+=self.energy_basis_1eMO[i,i]*2
        for i in range(Nelec2):
            for j in range(Nelec2):
                E+=2*self.energy_basis_2e_mol_chem[i,i,j,j]-self.energy_basis_2e_mol_chem[i,j,j,i]
        return E
    def tau_mat(self,ts,td):
    	tau_rel=td.copy()
    	tau_rel=tau_rel+np.einsum("ai,bj->abij",ts,ts)
    	tau_rel=tau_rel-np.einsum("bi,aj->abij",ts,ts)
    	return tau_rel
    def taus_mat(self,ts,td):
    	tausm_rel=td.copy()
    	tausm_rel=tausm_rel+np.einsum("ai,bj->abij",ts,ts)*0.5
    	tausm_rel=tausm_rel-np.einsum("bi,aj->abij",ts,ts)*0.5
    	return tausm_rel
    def updateintermediates(self,ts,td):
        Nelec, dim =self.Nelec, self.dim
        fs=self.fs
        spinints=self.spinints
        tau=self.tau_mat(ts,td)
        taus=self.taus_mat(ts,td)
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
    def T1_eq(self,ts,td,F):
        Nelec, dim =self.Nelec, self.dim
        fs=self.fs
        spinints=self.spinints
        Dai=self.Dai
        tsnew=np.einsum("ai->ia",fs[:Nelec,Nelec:dim],optimize=True)
        tsnew=tsnew+np.einsum("ei,ae->ai",ts,F[Nelec:dim,Nelec:dim],optimize=True)
        tsnew=tsnew-np.einsum("am,mi->ai",ts,F[:Nelec,:Nelec],optimize=True)
        tsnew=tsnew+np.einsum("aeim,me->ai",td,F[:Nelec,Nelec:dim],optimize=True)
        tsnew=tsnew-0.5*np.einsum("efim,maef->ai",td,spinints[:Nelec,Nelec:,Nelec:,Nelec:],optimize=True)
        tsnew=tsnew-0.5*np.einsum("aemn,nmei->ai",td,spinints[:Nelec,:Nelec,Nelec:dim,:Nelec],optimize=True)
        tsnew=tsnew-np.einsum("fn,naif->ai",ts,spinints[:Nelec,Nelec:dim,:Nelec,Nelec:dim],optimize=True)
        tsnew=tsnew-np.einsum("ai,ai->ai",ts,Dai[Nelec:dim,:Nelec],optimize=True)
        return tsnew
    def T1_eq_rhf(self,ts,td):
        pass

    def T2_eq_rhf(self,ts,td):
        pass
    def T2_eq(self,ts,td,F,W):
        Nelec, dim =self.Nelec, self.dim
        fs=self.fs
        spinints=self.spinints
        Dabij=self.Dabij
        tau=self.tau_mat(ts,td)
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
    def makeT1(self,ts,td,F):
        Nelec, dim =self.Nelec, self.dim
        fs=self.fs
        Dai=self.Dai
        spinints=self.spinints
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
    def makeT2(self,ts,td,F,W):
        Nelec, dim =self.Nelec, self.dim
        fs=self.fs
        Dabij=self.Dabij
        spinints=self.spinints
        tau=self.tau_mat(ts,td)
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
    def ccsdenergy(self,ts,td):
        Nelec, dim =self.Nelec, self.dim
        fs=self.fs
        spinints=self.spinints
        ECCSD = np.einsum("ia,ai->",fs[:Nelec,Nelec:dim],ts,optimize=True)
        ECCSD=ECCSD+0.25*np.einsum("ijab,abij->",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],td,optimize=True)
        ECCSD=ECCSD+0.5*np.einsum("ijab,ai,bj->",spinints[:Nelec,:Nelec,Nelec:dim,Nelec:dim],ts,ts,optimize=True)
        return ECCSD
    def setup(self):
        dim=self.dim
        Nelec=self.Nelec
        spinints=self.spinints
        fs=self.fs
        ts = np.zeros((dim-Nelec,Nelec))
        td = np.zeros((dim-Nelec,dim-Nelec,Nelec,Nelec))
        for a in range(0,dim-Nelec):
            for b in range(0,dim-Nelec):
                for i in range(0,Nelec):
                    for j in range(0,Nelec):
                        td[a,b,i,j] += spinints[i,j,a+Nelec,b+Nelec]/(fs[i,i] + fs[j,j] - fs[a+Nelec,a+Nelec] - fs[b+Nelec,b+Nelec])
        return ts,td
    def solve(self,ts=None,td=None):
        if ts is None:
            ts,td=self.setup()
        ECCSD = 0
        DECC = 1
        counter=0
        while DECC > 1e-8: # arbitrary convergence criteria
            OLDCC = ECCSD
            F,W = self.updateintermediates(ts,td)
            tsnew = self.makeT1(ts,td,F)
            tdnew = self.makeT2(ts,td,F,W)
            ts = tsnew
            td = tdnew
            ECCSD = self.ccsdenergy(ts,td)
            #print(DECC)
            DECC = abs(ECCSD - OLDCC)
            #t1_val=self.T1_eq(ts,td,F)
            #t2_val=self.T2_eq(ts,td,F,W)
            #print("Max T2-error",np.max(abs(t2_val)))
            #print("Max T1-error",np.max(abs(t1_val)))
            #print(DECC)
            counter+=1
        print("E(corr,CCSD) = ", ECCSD)
        print("E(CCSD) = ", ECCSD + self.ESCF)
        print("Number of convergence steps: %d"%counter)

#Step 1: Produce Fock matrix



def setUpsamples(sample_x,molecule,basis,rhf_mo_ref,mix_states=False,type="procrustes"):
    t1s=[]
    t2s=[]
    l1s=[]
    l2s=[]
    sample_energies=[]
    for x in sample_x:
        mol=make_mol(molecule,x,basis,charge=0)
        mf=scf.RHF(mol)
        ESCF=mf.kernel(verbose=0)
        if type=="basischange":
            rhf_mo=basischange(rhf_mo_ref,mol.intor("int1e_ovlp"),mol.nelectron//2)
        else:
            rhf_mo=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_mo_coeff=rhf_mo_ref,mix_states=mix_states)
        mf.mo_coeff=rhf_mo
        gmf=scf.addons.convert_to_ghf(mf)
        try:
            mycc = cc.CCSD(gmf)
            mycc.kernel()
        except:
            mycc = cc.CCSD(gmf)
            mycc.diis_start_cycle = 4
            mycc.diis_space = 10
            mycc.kernel()

        sample_energies.append(mycc.e_tot)
        t1_pyscf=mycc.t1
        t2_pyscf=mycc.t2
        l1_pyscf,l2_pyscf=mycc.solve_lambda()
        t1s.append(t1_pyscf.T)
        t2s.append(t2_pyscf.T)
        l1s.append(l1_pyscf.T)
        l2s.append(l2_pyscf.T)
    return t1s,t2s,l1s,l2s,sample_energies
def solve_evc(x_alphas,molecule,basis,rhf_mo_ref,t1s,t2s,l1s,l2s,mix_states=False,run_cc=True,cc_approx=True,type="procrustes"):
    """
    x_alphas: The sample geometries inpupt
    molecule: The function which returns the molecule
    basis: The basis set used
    rhf_mo_ref: The reference state
    t1s, t2s, l1s, l2s: The set of CC-coefficients to build the system with
    run_cc: Wether to solve the CC equations as well or not.
    cc_approx: Wether to get an approximative CC state or not
    """
    E_CCSD=[]
    E_approx=[]
    E_ownmethod=[]
    E_diffguess=[]
    E_RHF=[]
    for x_alpha in x_alphas:
        print(x_alpha)
        mol=make_mol(molecule,x_alpha,basis,charge=0)
        gccsolver=GCC_solver(mol,reference_state=rhf_mo_ref,mix_states=mix_states,type=type)
        ESCF=gccsolver.ESCF
        E_RHF.append(ESCF)
        if run_cc:
            mycc=cc.CCSD(gccsolver.gmf)
            try:
                mycc.kernel()
                mycc = cc.CCSD(gccsolver.gmf)
                E_CCSD.append(mycc.e_tot)
            except:
                mycc = cc.CCSD(gccsolver.gmf)
                mycc.diis_start_cycle = 4
                mycc.diis_space = 10
                mycc.kernel()
                E_CCSD.append(mycc.e_tot)
        H=np.zeros((len(t1s),len(t1s)))
        S=np.zeros((len(t1s),len(t1s)))
        for i, xi in enumerate(t1s):
            F,W = gccsolver.updateintermediates(t1s[i],t2s[i])
            t1_error=gccsolver.T1_eq(t1s[i],t2s[i],F)
            t2_error=gccsolver.T2_eq(t1s[i],t2s[i],F,W)
            for j, xj in enumerate(t1s):
                X1=t1s[i]-t1s[j]
                X2=t2s[i]-t2s[j]
                overlap=0
                overlap+=1+np.einsum("ia,ai->",l1s[j].T,X1,optimize=True)
                overlap+=0.5*np.einsum("ijab,ai,bj->",l2s[j].T,X1,X1,optimize=True)+0.25*np.einsum("ijab,abij->",l2s[j].T,X2,optimize=True)
                #overlap=1+np.einsum("ia,ai->",l1s[j],X1)+0.5*contract("ijab,ai,bj->",self.l2s[j],X1,X1)+0.25*contract("ijab,abij->",self.l2s[j],X2)
                S[i,j]=overlap
                exp_energy=gccsolver.ccsdenergy(t1s[i],t2s[i])+ESCF
                H[i,j]=overlap*exp_energy
                extra=np.einsum("ia,ai->",l1s[j].T,t1_error,optimize=True)+np.einsum("ijab,ai,bj->",l2s[j].T,X1,t1_error,optimize=True)+0.25*np.einsum("ijab,abij->",l2s[j].T,t2_error,optimize=True)
                H[i,j]=H[i,j]+extra
        print(S)
        print(H)
        sys.exit(1)
        e,cl,c=eig(scipy.linalg.inv(S+np.eye(len(S))*1e-8)@H,left=True)

        idx = np.real(e).argsort()
        e = e[idx]
        c = c[:,idx]
        cl = cl[:,idx]
        E_approx.append(np.real(e[0]))
        if cc_approx:
            t1own,t2own=best_CC_fit(t1s,t2s,np.real(c[:,0])/np.sum(np.real(c[:,0])))
            own_energy=gccsolver.ccsdenergy(t1own,t2own)+ESCF
            E_ownmethod.append(own_energy)
    return E_CCSD,E_approx,E_diffguess,E_RHF,E_ownmethod
def solve_evc2(x_alphas,molecule,basis,rhf_mo_ref,t1s,t2s,l1s,l2s,mix_states=False,type="procrustes"):
    def error_function(params,gccsolver,t1s,t2s,jac=False):
        t1=np.zeros(t1s[0].shape)
        t2=np.zeros(t2s[0].shape)
        for i in range(len(t1s)):
            t1+=params[i]*t1s[i] #Starting guess
            t2+=params[i]*t2s[i] #Starting guess
        projection_errors=np.zeros(len(t1s))
        F,W = gccsolver.updateintermediates(t1,t2)
        t1_error=gccsolver.T1_eq(t1,t2,F)
        t2_error=gccsolver.T2_eq(t1,t2,F,W)
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
    energy=[]
    start_guess=np.full(len(t1s),1/len(t1s))
    t1s,t2s=orthonormalize_ts(t1s,t2s)
    for x_alpha in x_alphas:
        mol=make_mol(molecule,x_alpha,basis,charge=0)
        gccsolver=GCC_solver(mol,reference_state=rhf_mo_ref,mix_states=mix_states,type=type)
        ESCF=gccsolver.ESCF
        jacob=False
        sol=root(error_function,start_guess,args=(gccsolver,t1s,t2s,jacob),jac=jacob,method="hybr",options={"xtol":1e-3})#,method="broyden1")
        try:
            print("Converged: ",sol.success, " number of iterations:",sol.nit)
        except:
            print("Converged: ",sol.success, " number of iterations:",sol.nfev)
        final=sol.x
        t1=np.zeros(t1s[0].shape)
        t2=np.zeros(t2s[0].shape)
        for i in range(len(t1s)):
            t1+=final[i]*t1s[i] #Starting guess
            t2+=final[i]*t2s[i] #Starting guess
        newEn=gccsolver.ccsdenergy(t1,t2)+ESCF
        energy.append(newEn)
    return energy
if __name__=="__main__":
    #molecule=lambda x: "Be 0 0 0; H 0 0 %f; H 0 0 %f"%(x,-x)
    molecule=lambda x: "F 0 0 %f; H 0 0 0"%(x)
    #molecule=lambda x: "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
    mix_states=False
    basis="6-31G"
    molecule_name="HF"
    sample_geom1=np.linspace(2,3,2)

    sample_geom=[[x] for x in sample_geom1]
    sample_geom1=np.array(sample_geom).flatten()
    geom_alphas1=np.linspace(1,4,31)
    geom_alphas=[[x] for x in geom_alphas1]
    #molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))

    ref_x=2
    type="procrustes"
    mol=make_mol(molecule,[ref_x],basis,charge=0)
    ENUC=mol.energy_nuc()
    mf=scf.RHF(mol)
    mf.kernel()
    rhf_mo_ref=mf.mo_coeff
    fig, ax1 = plt.subplots(1,sharex=True)
    #E_CCSD=CC_energy_curve(np.array(geom_alphas).ravel(),basis,molecule,unit="Bohr")
    #E_FCI=FCI_energy_curve(np.array(geom_alphas).ravel(),basis,molecule,unit="Bohr")
    t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,rhf_mo_ref,mix_states=mix_states,type=type)
    t2_test=t2s[0]


    #x_alphas=np.linspace(0,4,41)
    E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis,rhf_mo_ref,t1s,t2s,l1s,l2s,run_cc=False,cc_approx=True,mix_states=mix_states,type=type)

    #energy_simen=solve_evc2(geom_alphas,molecule,basis,rhf_mo_ref,t1s,t2s,l1s,l2s,mix_states=mix_states,type=type)
    #ax1.plot(geom_alphas1,E_RHF,label="RHF")
    #ax1.plot(geom_alphas1,E_CCSD,label="CCSD")
    #ax1.plot(geom_alphas1,E_CCSDx,label="CCSD")
    ax1.plot(geom_alphas1,E_approx,label="EVC (approach 1)")
    #ax1.plot(geom_alphas1,E_ownmethod,label="Fit to approach 1")
    #ax1.plot(geom_alphas1,energy_simen,label="EVC (approach 2)")
    ax1.plot(sample_geom1,sample_energies,"*",label="Sampling points")
    #ax1.plot(geom_alphas1,E_FCI,label="Full CI")
    ax1.set_ylabel("Energy (Hartree)")
    ax1.legend()
    #ax2.plot(geom_alphas1,abs(E_RHF-E_FCI),label="RHF")
    #ax2.plot(geom_alphas1,abs(E_CCSD-E_FCI),label="CCSD")
    #ax2.plot(geom_alphas1,abs(E_approx-E_FCI),label="EVC (approach 1)")
    #ax2.plot(geom_alphas1,abs(E_ownmethod-E_FCI),label="Fit to approach 1")
    #ax2.plot(geom_alphas1,abs(energy_simen-E_FCI),label="EVC (approach 2)")
    #ax2.set_xlabel("Distance (Bohr)")
    #ax2.set_ylabel("Energy (Hartree)")
    #ax2.set_yscale("log")
    #ax2.legend()
    plt.tight_layout()
    #plt.savefig("%s_%s_%d.pdf"%(molecule_name,basis,len(sample_x)))
    plt.show()
