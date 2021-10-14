import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf,ao2mo,lo
import scipy
import sys
from numba import jit
sys.path.append("../eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *
np.set_printoptions(linewidth=200,precision=5,suppress=True)
def swap_cols(arr, frm, to):
    """Swaps the columns of a 2D-array"""
    arrny=arr.copy()
    arrny[:,[frm, to]] = arrny[:,[to, frm]]
    return arrny
def swappistan(matrix):
    swapperinos=[]
    for i in range((matrix.shape[1])):
        for j in range(i+1,matrix.shape[1]):
            sort_i=np.sort(matrix[:,i])
            sort_j=np.sort(matrix[:,j])
            if(np.all(np.abs(sort_i-sort_j)<1e-8)): #If the two columns are equal
                nonzero_i=np.where(np.abs(matrix[:,i])>=1e-5)[0][0]
                nonzero_j=np.where(np.abs(matrix[:,j])>=1e-5)[0][0]
                if nonzero_i>nonzero_j:
                    matrix=swap_cols(matrix,i,j)
    return matrix
def cholesky_pivoting(matrix):
    n=len(matrix)
    R=np.zeros((n,n))
    piv=np.arange(n)
    for k in range(n):
        q=np.argmax(np.diag(matrix)[k:])+k
        if matrix[q,q]<1e-14:
            break
        temp=matrix[:,k].copy()
        matrix[:,k]=matrix[:,q]
        matrix[:,q]=temp
        temp=R[:,k].copy()
        R[:,k]=R[:,q]
        R[:,q]=temp
        temp=matrix[k,:].copy()
        matrix[k,:]=matrix[q,:]
        matrix[q,:]=temp
        temp=piv[k]
        piv[k]=piv[q]
        piv[q]=temp
        R[k,k]=np.sqrt(matrix[k,k])
        R[k,k+1:]=matrix[k,k+1:]/R[k,k]
        matrix[k+1:n,k+1:n]=matrix[k+1:n,k+1:n]-np.outer(R[k,k+1:],R[k,k+1:])
    P=np.eye(n)[:,piv]
    return R,P
def cholesky_coefficientmatrix(matrix):
    D=2*matrix@matrix.T
    R,P=cholesky_pivoting(D)
    PL=P@R.T
    Cnew=PL[:,:matrix.shape[1]]/np.sqrt(2)
    return Cnew
def localize_mocoeff(mol,mo_coeff,mo_occ):
    """
    mo = lo.ER(mol, mo_coeff[:,mo_occ>0])
    mo.init_guess = None
    mo = mo.kernel()
    mo=swappistan(mo)
    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo = lo.ER(mol, mo_coeff[:,mo_occ<=0])
    mo.init_guess = None
    mo = mo.kernel()
    mo=swappistan(mo)
    mo_coeff[:,mo_occ<=0]=np.array(mo)
    """
    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ>0])
    mo=swappistan(mo)

    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ<=0])
    mo=swappistan(mo)

    mo_coeff[:,mo_occ<=0]=np.array(mo)

    return mo_coeff

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
        diffloc=np.where(bra.alpha!=ket.alpha)[0][0] #The index where the elements are different
        return onebody[bra.alpha[diffloc],ket.alpha[diffloc]] #Return the energy at that point
    elif diff_beta==1:
        diffloc=np.where(bra.beta!=ket.beta)[0][0]
        return onebody[bra.beta[diffloc],ket.beta[diffloc]]
    else:
        energy=0
        for i in bra.alpha: #bra.alpha is equal to ket.alpha
            energy+=onebody[i,i]
        for i in bra.beta:  #bra.beta is equal to ket.beta
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
def e_doublesum_1(bra,ket,twobody):
    energy=0
    for m in bra:
        for n in ket:
            energy+=twobody[m,m,n,n]
def e_doublesum_2(bra,ket,twobody):
    energy=0
    for m in bra:
        for n in ket:
            energy-=twobody[m,n,n,m]
class RHF_CISDsolver(): #Closed-shell only
    def __init__(self,mol):
        self.mol=mol
        mf=scf.RHF(mol)
        mf.kernel()
        self.onebody=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
        self.twobody=mol.intor('int2e',aosym="s1")
        self.overlap=mol.intor("int1e_ovlp")
        self.mo_coeff=localize_mocoeff(mol,mf.mo_coeff,mf.mo_occ)
        print(self.mo_coeff)
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
                    T[i,j]+=self.mol.energy_nuc() #Add nuc. repulsion to diagonal
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
        lowest_eigenvector=lowest_eigenvector*np.sign(lowest_eigenvector[0])
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
        if self.T is None:
            self.setupT()
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
        for i in range(self.n_occ):
            for a in range(self.n_occ,self.num_bas):
                beta1=np.arange(self.neh)
                beta1[i]=a #Replace i with a
                wf=permutation_wf(1)
                wf.add_config(beta1,beta1)
                wfs.append(wf)
        return wfs
    def create_doubles_samediff(self):
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
def create_sample_coefficients(molecule,x_sample):
    coefficients=[]
    energies=[]
    for x in x_sample:
        mol=gto.Mole()
        mol.atom=molecule(x)
        mol.basis = '6-31G'
        mol.unit= "Bohr"
        mol.build()
        solverino=RHF_CISDsolver(mol)
        e,coeff,T=solverino.calculate_coefficients()
        coefficients.append(coeff)
        energies.append(e)
    return coefficients,energies


def eigvec(c_arr,x_arr,molecule,printout=False):
    energies=np.empty(len(x_arr))
    energies_reference=np.empty(len(x_arr))
    T=np.empty((len(c_arr),len(c_arr)))
    S=np.empty((len(c_arr),len(c_arr)))
    for index, x in enumerate(x_arr):
        mol=gto.Mole()
        mol.atom=molecule(x)
        mol.basis = '6-31G'
        mol.unit= "Bohr"
        mol.build()
        solverino=RHF_CISDsolver(mol)
        eref,coeff,soppel=solverino.calculate_coefficients() #Also sets up the T matrix
        for i in range(len(c_arr)):
            coeff_left=c_arr[i]
            for j in range(i,len(c_arr)):
                coeff_right=c_arr[j]
                T[i,j]=T[j,i]=solverino.calculate_T_entry(coeff_left,coeff_right)
                S[i,j]=S[j,i]=solverino.calculate_overlap(coeff_left,coeff_right)
        e,vec=generalized_eigenvector(T,S)
        energies[index]=e
        energies_reference[index]=eref

    return energies, energies_reference


molecule=lambda x: "H 0 0 0; F 0 0 %f"%x
"""
mol=gto.Mole()
mol.atom=molecule(1.2)
mol.basis = '6-31G'
mol.unit= "Bohr"
mol.build()
solverino=RHF_CISDsolver(mol)
eref,coeff,T_1=solverino.calculate_coefficients()
print(eref)
mol=gto.Mole()
mol.atom=molecule(1.3)
mol.basis = '6-31G'
mol.unit= "Bohr"
mol.build()
solverino=RHF_CISDsolver(mol)
eref,coeff1,T_2=solverino.calculate_coefficients()
print(eref)
print(np.max(np.abs(coeff1-coeff)))
"""
sample_x=np.linspace(1,6,6)
plot_x=np.linspace(1,6,21)
cisd=CISD_energy_curve(plot_x,"6-31G",molecule)
hf_curve=energy_curve_RHF(plot_x,"6-31G",molecule)

coeffs,sample_energies=create_sample_coefficients(molecule,sample_x)
for c in coeffs:
    print(c)

energies,energies_reference=eigvec(coeffs,plot_x,molecule,printout=False)
energies_atsamplepoints=eigvec(coeffs,sample_x,molecule,printout=True)
plt.plot(sample_x,sample_energies,"*",label="samples")
plt.plot(plot_x,cisd,label="pyscf")
plt.plot(plot_x,hf_curve,label="Hartree-Fock")

plt.plot(plot_x,energies,label="EVC")
plt.plot(plot_x,energies_reference,label="shitty CISD")
plt.legend()
plt.show()
print(energies_reference)
