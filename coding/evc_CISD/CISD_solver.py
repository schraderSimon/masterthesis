import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf,ao2mo
import scipy
import sys
sys.path.append("../eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *


class SL_det():
    def __init__(self,alpha_string,beta_string):
        self.alpha=alpha_string
        self.beta=beta_string
class permutation_wf():
    def __init__(self,number_determinants):
        self.prefactor=1/np.sqrt(number_determinants)
        self.determinants=[]
    def add_determinant(self,determinant):
        self.determinants.append(determinant)
    def add_config(self,alpha_string,beta_string):
        self.determinants.append(SL_det(alpha_string,beta_string))
def energy_between_wfs(bra,ket,onebody,twobody):
    e1=0
    e2=0
    prefac=bra.prefactor*ket.prefactor
    for bra_SL in bra.determinants:
        for ket_SL in ket.determinants:
            e1+=prefac*calculate_e1(bra_SL,ket_SL,onebody)
            e2+=prefac*calculate_e2(bra_SL,ket_SL,twobody)
    return e1+e2
def calculate_e1(bra,ket,onebody):
    diff_alpha=np.sum(bra.alpha!=ket.alpha) #Number of different alpha elements
    diff_beta=np.sum(bra.beta!=ket.beta)
    if diff_alpha+diff_beta>1:
        return 0
    elif diff_alpha==1:
        diffloc=np.where(bra.alpha!=ket.alpha)[0][0]
        return onebody[bra.alpha[diffloc],ket.alpha[diffloc]]
    elif diff_beta==1:
        diffloc=np.where(bra.beta!=ket.beta)[0][0]
        return onebody[bra.beta[diffloc],ket.beta[diffloc]]
    else:
        energy=0
        for i in bra.alpha:
            energy+=onebody[i,i]
        for i in bra.beta:
            energy+=onebody[i,i]
        return energy
def calculate_e2(bra,ket,twobody):
    diff_alpha=np.sum(bra.alpha!=ket.alpha) #Number of different alpha elements
    diff_beta=np.sum(bra.beta!=ket.beta)
    if diff_alpha+diff_beta>2:
        return 0
    elif diff_alpha==2:
        diffloc=np.where(bra.alpha!=ket.alpha)[0]
        m=bra.alpha[diffloc[0]]
        n=bra.alpha[diffloc[1]]
        p=ket.alpha[diffloc[0]]
        q=ket.alpha[diffloc[1]]
        return twobody[m,p,n,q]-twobody[m,q,n,p]
    elif diff_beta==2:
        diffloc=np.where(bra.beta!=ket.beta)[0]
        m=bra.beta[diffloc[0]]
        n=bra.beta[diffloc[1]]
        p=ket.beta[diffloc[0]]
        q=ket.beta[diffloc[1]]
        return twobody[m,p,n,q]-twobody[m,q,n,p]
    elif diff_alpha==1 and diff_beta==1:
        diffloc_alpha=np.where(bra.alpha!=ket.alpha)[0]
        diffloc_beta=np.where(bra.beta!=ket.beta)[0]
        m=bra.alpha[diffloc_alpha[0]]
        n=bra.beta[diffloc_beta[0]]
        p=ket.alpha[diffloc_alpha[0]]
        q=ket.beta[diffloc_beta[0]]
        return twobody[m,p,n,q]
    elif diff_alpha==1:
        diffloc_alpha=np.where(bra.alpha!=ket.alpha)[0]
        m=bra.alpha[diffloc_alpha[0]]
        p=ket.alpha[diffloc_alpha[0]]
        energy=0
        for n in bra.alpha:
            energy+=twobody[m,p,n,n]-twobody[m,n,n,p]
        for n in bra.beta:
            energy+=twobody[m,p,n,n]
        return energy
    elif diff_beta==1:
        diffloc_beta=np.where(bra.beta!=ket.beta)[0]
        m=bra.beta[diffloc_beta[0]]
        p=ket.beta[diffloc_beta[0]]
        energy=0
        for n in bra.alpha:
            energy+=twobody[m,p,n,n]
        for n in bra.beta:
            energy+=twobody[m,p,n,n]-twobody[m,n,n,p]
        return energy
    else:
        energy=0
        for m in bra.alpha:
            for n in ket.alpha:
                energy+=twobody[m,m,n,n]-twobody[m,n,n,m]
        for m in bra.alpha:
            for n in ket.beta:
                energy+=twobody[m,m,n,n]
        for m in bra.beta:
            for n in ket.alpha:
                energy+=twobody[m,m,n,n]
        for m in bra.beta:
            for n in ket.beta:
                energy+=twobody[m,m,n,n]-twobody[m,n,n,m]
        return 0.5*energy
class RHF_CISDsolver(): #Closed-shell only
    def __init__(self,mol):
        self.mol=mol
        if mol.spin ==0:
            mf=scf.RHF(mol)
            print(mf.kernel())
        self.onebody=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
        self.twobody=mol.intor('int2e',aosym="s1")
        self.overlap=mol.intor("int1e_ovlp")
        self.mo_coeff=mf.mo_coeff
        self.onebody=np.einsum("ki,lj,kl->ij",self.mo_coeff,self.mo_coeff,mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
        self.twobody=ao2mo.get_mo_eri(mol.intor('int2e',aosym="s1"),(self.mo_coeff,self.mo_coeff,self.mo_coeff,self.mo_coeff),aosym="s1")
        self.ne=mol.nelectron
        self.neh=int(self.ne/2)
        self.n_occ=self.neh
        self.num_bas=(self.mo_coeff.shape[0])
        self.n_unocc=self.num_bas-self.n_occ
        self.doubles=True
        self.energy=None
        self.coeff=None
        self.T=None
    def setupT(self):
        all_wfs=self.create_GS()+self.create_singles()+self.create_doubles_samesame()+self.create_doubles_samediff()+self.create_doubles_diffsame()+self.create_doubles_diffdiff()
        T=np.empty((len(all_wfs),len(all_wfs)))
        for i in range(len(T)):
            for j in range(i,len(T)):
                T[j,i]=T[i,j]=energy_between_wfs(all_wfs[i],all_wfs[j],self.onebody,self.twobody)
                if (i==j):
                    T[i,j]+=self.mol.energy_nuc()
        self.T=T
    def calculate_coefficients(self):
        if self.T is None:
            self.setupT()
        energy,eigenvector=np.linalg.eigh(self.T)
        idx = energy.argsort()[::1] #Order by size (non-absolute)
        energy = energy[idx]
        eigenvector = eigenvector[:,idx]
        lowest_eigenvalue=energy[0]
        lowest_eigenvector=eigenvector[:,0]
        self.coeff=lowest_eigenvector
        return lowest_eigenvalue,lowest_eigenvector, self.T
    def calculate_energy(self,coeff=None,T=None):
        if coeff is None:
            coeff=self.coeff
            T=self.T
        return coeff.T@T@coeff
    def calculate_T_entry(self,coeff_left,coeff_right,T=None):
        if T is None:
            T=self.T
        return coeff_left.T@T@coeff_right
    def calculate_overlap(self,coeff_left,coeff_right):
        return coeff_left.T@coeff_right
    def change_molecule(self,mol):
        self.__init__(mol)
    def create_GS(self):
        wfs=[]
        wf=permutation_wf(1)
        wf.add_config(np.arange(self.neh),np.arange(self.neh))
        wfs.append(wf)
        return wfs
    def create_singles(self):
        wfs=[]
        #1. Create all possible permutations
        print("Singles")
        for i in range(self.n_occ):
            for j in range(self.n_occ,self.num_bas):
                alpha1=np.arange(self.neh)
                beta1=np.arange(self.neh)
                beta1[i]=j #Replace i with j
                wf=permutation_wf(2)
                wf.add_config(alpha1,beta1)
                wf.add_config(beta1,alpha1)
                wfs.append(wf)
        return wfs
    def create_doubles_samesame(self):
        wfs=[]
        print("Doubles_samsame")
        for i in range(self.n_occ):
            for a in range(self.n_occ,self.num_bas):
                beta1=np.arange(self.neh)
                beta1[i]=a #Replace i with a
                wf=permutation_wf(1)
                wf.add_config(beta1,beta1)
                wfs.append(wf)
        return wfs
    def create_doubles_samediff(self):
        print("Doubles samediff")
        wfs=[]
        for i in range(self.n_occ):
            for a in range(self.n_occ,self.num_bas):
                for b in range(a+1,self.num_bas):
                    alpha1=np.arange(self.neh)
                    alpha2=np.arange(self.neh)
                    beta1=np.arange(self.neh)
                    beta2=np.arange(self.neh)
                    alpha1[i]=a
                    alpha2[i]=b
                    beta1[i]=b
                    beta2[i]=a
                    wf=permutation_wf(2)
                    wf.add_config(alpha1,beta1)
                    wf.add_config(alpha2,beta2)
                    wfs.append(wf)
        return wfs
    def create_doubles_diffsame(self):
        wfs=[]
        print("Diffsame")
        for i in range(self.n_occ):
            for j in range(i+1,self.n_occ):
                for a in range(self.n_occ,self.num_bas):
                    alpha1=np.arange(self.neh)
                    alpha2=np.arange(self.neh)
                    beta1=np.arange(self.neh)
                    beta2=np.arange(self.neh)
                    alpha1[i]=a
                    alpha2[j]=a
                    beta1[j]=a
                    beta2[i]=a
                    wf=permutation_wf(2)
                    wf.add_config(alpha1,beta1)
                    wf.add_config(alpha2,beta2)
                    wfs.append(wf)
        return wfs
    def create_doubles_diffdiff(self):
        wfs=[]
        print("Diffdiff")
        for i in range(self.n_occ):
            for j in range(self.n_occ):
                if(j==i):
                    continue
                for a in range(self.n_occ,self.num_bas):
                    for b in range(a+1,self.num_bas):
                        if(a==b):
                            continue
                        alpha1=np.arange(self.neh)
                        alpha2=np.arange(self.neh)
                        beta1=np.arange(self.neh)
                        beta2=np.arange(self.neh)
                        alpha3=np.arange(self.neh)
                        alpha4=np.arange(self.neh)
                        beta3=np.arange(self.neh)
                        beta4=np.arange(self.neh)
                        alpha1[i]=a; alpha1[j]=b
                        alpha2[i]=a; beta2[j]=b
                        alpha3[j]=b; beta3[i]=a
                        beta4[i]=a; beta4[j]=b
                        wf=permutation_wf(4)
                        wf.add_config(alpha1,beta1)
                        wf.add_config(alpha2,beta2)
                        wf.add_config(alpha3,beta4)
                        wf.add_config(alpha4,beta4)
                        wfs.append(wf)
        return wfs
mol=gto.Mole()
mol.atom = 'F 0 0 0; H 0 1.5 0'
mol.basis = '6-31G'
mol.unit= "Bohr"
mol.build()
solverino=RHF_CISDsolver(mol)
energy1,coeff1,T1=solverino.calculate_coefficients()

mol=gto.Mole()
mol.atom = 'F 0 0 0; H 0 2.0 0'
mol.basis = '6-31G'
mol.unit= "Bohr"
mol.build()
solverino=RHF_CISDsolver(mol)
energy2,coeff2,T2=solverino.calculate_coefficients()

mol=gto.Mole()
mol.atom = 'F 0 0 0; H 0 1.75 0'
mol.basis = '6-31G'
mol.unit= "Bohr"
mol.build()
solverino=RHF_CISDsolver(mol)
energy4,coeff4,T4=solverino.calculate_coefficients()


mol=gto.Mole()
mol.atom = 'F 0 0 0; H 0 2.5 0'
mol.basis = '6-31G'
mol.unit= "Bohr"
mol.build()
solverino=RHF_CISDsolver(mol)
energy3,coeff3,T3=solverino.calculate_coefficients()
print("Bad solver: %f"%energy3)

c_arr=[coeff1,coeff2,coeff4]

mf=mol.HF.run()
mf.run()
mycisd=mf.CISD().run()

def eigvec(c_arr,solverino):
    T=np.empty((len(c_arr),len(c_arr)))
    S=np.empty((len(c_arr),len(c_arr)))
    for i in range(len(c_arr)):
        coeff_left=c_arr[i]
        for j in range(i,len(c_arr)):
            coeff_right=c_arr[j]
            T[i,j]=T[j,i]=solverino.calculate_T_entry(coeff_left,coeff_right)
            S[i,j]=S[j,i]=solverino.calculate_overlap(coeff_left,coeff_right)
    e,vec=generalized_eigenvector(T,S)
    print("Eigvec: %f"%e)
eigvec(c_arr,solverino)


"""

class UHF_CISDsolver(): #Closed-shell only
    def __init__(self,mol):
        self.mol=mol
        if mol.spin==0:
            mf=scf.RHF(mol)
            mf.kernel()
        self.onebody=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
        self.twobody=mol.intor('int2e',aosym="s1")
        self.overlap=mol.intor("int1e_ovlp")
        self.mo_coeff=mf.mo_coeff
        self.onebody=np.einsum("ki,lj,kl->ij",self.mo_coeff,self.mo_coeff,mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
        self.twobody=ao2mo.get_mo_eri(mol.intor('int2e',aosym="s1"),(self.mo_coeff,self.mo_coeff,self.mo_coeff,self.mo_coeff),aosym="s1")
        self.ne=mol.nelectron
        self.neh=int(self.ne/2)
        self.n_occ=self.ne
        self.num_bas=(self.mo_coeff.shape[0])
        self.n_unocc=self.num_bas-self.nOCC
        self.doubles=True
        self.E_elements=None
        self.coefficients=None
    def calculate_coefficients(self,mol):
        c0=0
        c1=np.empty((self.neh*2,self.num_bas*2)) #Single excitation coefficients
        c2=np.empty((self.neh*2,self.neh*2,self.num_bas*2,self.num_bas*2)) #Double excitation coefficients
        total=1+len(c1.ravel())+len(c2.ravel(()))
        Tarr=np.zeros((total,total))
        GS_coefficients=np.arange(self.neh)
        singlet_coefficients=[]
        single_permutations=create_singles()
        double_permutations=create_doubles()
        for perm in range(len(single_permutations)):
            temp=np.arange(self.neh)
            temp[perm[0]]=perm[1]
            singlet_coefficients.append(temp)
        for j in range(len(double_permutations)):
            temp=np.arange(self.neh)
            temp=np.arange(self.neh)
    def change_molecule(self,mol):
        self.__init__(mol)
    def create_singles(self):
        permutations=[]
        #1. Create all possible permutations
        for i in range(self.n_occ):
            for j in range(self.n_occ,self.num_bas):
                permutations.append([[i,j]) #This means: i out, j in!. The zero is there as a "do nothing" operator (now we are operating two permutations...)
        return permutations
    def create_doubles(self):
        if self.doubles==False:
            return []
        permutations=[]
        for j in range(self.n_occ*2):
            for i in range(self.n_occ*2): #j
                for a in range(self.n_occ*2,self.num_bas*2):
                    for b in range(self.n_occ*2,self.num_bas*2): #l
                        permutations.append([[i,j],[a,b]])
"""
