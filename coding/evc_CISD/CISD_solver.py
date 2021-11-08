import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf,ao2mo,lo,ci,fci,cc
import scipy
import sys
from numba import jit
import scipy
sys.path.append("../eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *
from scipy.optimize import minimize, minimize_scalar
#from scipy.linalg import orthogonal_procrustes
np.set_printoptions(linewidth=200,precision=5,suppress=True)
def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo.T
    B=mo_new.T
    M=B@A.T
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0

def similarize_1(MO_tochange,MO_ref,f,dev):
    num_bas=len(MO_tochange[0,:])
    for i in range(num_bas):
        for j in range(i+1,num_bas):
            x1=MO_ref[:,i]
            x2=MO_ref[:,j]
            y1=MO_tochange[:,i]
            y2=MO_tochange[:,j]
            before=f(0,y1,y2,x1,x2)
            alpha=minimize_scalar(f,bounds=(0,2*np.pi),args=(y1,y2,x1,x2)).x#[0]#,jac=dev,method="BFGS").x[0]#,jac=derivative_of_function).x[0]
            after=f(alpha,y1,y2,x1,x2)
            sa=np.sin(alpha)
            ca=np.cos(alpha)
            y1_new=ca*MO_tochange[:,i]-sa*MO_tochange[:,j]
            y2_new=sa*MO_tochange[:,i]+ca*MO_tochange[:,j]
            MO_tochange[:,i]=y1_new
            MO_tochange[:,j]=y2_new
    return MO_tochange
def similarize(MO_tochange,MO_ref,f,dev):
    num_bas=len(MO_tochange[0,:])
    for i in range(num_bas):
        for j in range(i+1,num_bas):
            x1=MO_ref[:,i]
            x2=MO_ref[:,j]
            y1=MO_tochange[:,i]
            y2=MO_tochange[:,j]
            power=1
            before=np.sum(np.abs(y1-x1)**power)+np.sum(np.abs(y2-x2)**power)
            alpha=minimize_scalar(f,bounds=(0,2*np.pi),args=(y1,y2,x1,x2)).x#[0]#,jac=dev,method="BFGS").x[0]#,jac=derivative_of_function).x[0]
            sa=np.sin(alpha)
            ca=np.cos(alpha)
            y1_new=(ca*y1+sa*y2)
            y2_new=(sa*y1-ca*y2)
            after=np.sum(np.abs(y1_new-x1)**power)+np.sum(np.abs(y2_new-x2)**power)
            if before<after:
                MO_tochange[:,i]=y1
                MO_tochange[:,j]=y2
            else:
                MO_tochange[:,i]=y1_new
                MO_tochange[:,j]=y2_new
    return MO_tochange

def orbital_dissimilarity(alpha,y1,y2,x1,x2,printerino=False):
    ca=np.cos(alpha)
    sa=np.sin(alpha)
    power=1
    first=np.sum(np.abs(ca*y1+sa*y2-x1)**power)
    second=np.sum(np.abs(sa*y1-ca*y2-x2)**power)
    if printerino:
        print(y1,ca*y1+sa*y2)
        print(y2,sa*y1-ca*y2)
    return first+second
def orbital_dissimilarity_dev(alpha,y1,y2,x1,x2):
    ca=np.cos(alpha)
    sa=np.sin(alpha)
    first=np.sum((ca*y1+sa*y2-x1)*(-sa*y1+ca*y2))
    second=np.sum((sa*y1-ca*y2-x2)*((ca*y1+sa*y2)))
    return first+second
def orbital_dissimilarity_1(alpha,y1,y2,x1,x2):
    ca=np.cos(alpha)
    sa=np.sin(alpha)
    power=1
    first=np.sum(np.abs(ca*y1-sa*y2-x1)**power)
    second=np.sum(np.abs(sa*y1+ca*y2-x2)**power)
    return first+second
def orbital_dissimilarity_dev_1(alpha,y1,y2,x1,x2):
    ca=np.cos(alpha)
    sa=np.sin(alpha)
    first=np.sum((ca*y1-sa*y2-x1)*(-sa*y1-ca*y2))
    second=np.sum((sa*y1+ca*y2-x2)*((ca*y1-sa*y2)))
    return first+second

def localize_cholesky(mol,mo_coeff,mo_occ):
    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ>0])
    mo=swappistan(mo)
    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo_unocc=cholesky_coefficientmatrix(mo_coeff[:,mo_occ<=0])
    mo_unocc=swappistan(mo_unocc)
    mo_coeff[:,mo_occ<=0]=np.array(mo_unocc)
    return mo_coeff
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

def calc_diff_bitstring(a,b):
    difference_alpha=int(a[:len(a)//2],2)^int(b[:len(b)//2],2)
    difference_beta=int(a[len(a)//2:],2)^int(b[len(b)//2:],2)
    ndiffa=bin(difference_alpha).count("1") #How many differ
    ndiffb=bin(difference_beta).count("1") #How many differ
    return ndiffa,ndiffb

class RHF_CISDsolver():
    def __init__(self,mol,mo_coeff=None,basischange=False,nonzeros=None):
        self.mol=mol
        self.overlap=mol.intor("int1e_ovlp")
        if mo_coeff is None:
            #If no parameter is given, we simply use the cholesky coefficient matrices
            print("Finding a new SL-det")
            mf=scf.RHF(mol)
            mf.kernel()
            self.mo_coeff=localize_cholesky(mol,mf.mo_coeff,mf.mo_occ)
        else:
            #If a given basis is given, then we either basis change it, or not.
            if basischange:
                self.mo_coeff=self.basischange(mo_coeff,self.overlap)
            else:
                self.mo_coeff=mo_coeff
        self.onebody=np.einsum("ki,lj,kl->ij",self.mo_coeff,self.mo_coeff,mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
        self.twobody=ao2mo.get_mo_eri(mol.intor('int2e',aosym="s1"),(self.mo_coeff,self.mo_coeff,self.mo_coeff,self.mo_coeff),aosym="s1")
        self.ne=mol.nelectron
        self.neh=int(self.ne/2)
        self.n_occ=self.neh
        self.num_bas=(self.mo_coeff.shape[0])
        self.n_unocc=self.num_bas-self.n_occ
        self.energy=None
        self.coeff=None
        self.T=None
        self.nonzeros=nonzeros
        self.data_creator()
    def getdeterminant_matrix(self,AO_overlap,HF_coefficients_left,HF_coefficients_right):
        determinant_matrix=np.einsum("ab,ai,bj->ij",AO_overlap,HF_coefficients_left,HF_coefficients_right)
        return determinant_matrix
    def basischange(self,C_old,overlap_AOs_newnew):
        S=np.einsum("mi,vj,mv->ij",C_old,C_old,overlap_AOs_newnew)
        S_eig,S_U=np.linalg.eigh(S)
        S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
        C_new=np.einsum("ij,mj->mi",S_powerminusonehalf,C_old)
        return C_new
    def data_creator(self):
        neh=self.neh
        num_bas=self.num_bas
        states=self.state_creator()
        if self.nonzeros is not None:
            self.states=[states[i] for i in self.nonzeros]
        else:
            self.states=states
        states_fallen=[state[0]+state[1] for state in self.states]
        indices=np.array(self.index_creator())
        self.num_states=len(self.states)
        self.states_fallen=states_fallen
        self.indices=indices

    def state_creator(self):
        neh=self.neh
        groundstring="1"*neh+"0"*(self.n_unocc) #Ground state Slater determinant
        alpha_singleexcitations=[]
        for i in range(neh):
            for j in range(neh,self.num_bas):
                newstring=groundstring[:i]+"0"+groundstring[i+1:j]+"1"+groundstring[j+1:] #Take the ith 1 and move it to position j
                alpha_singleexcitations.append(newstring)
        alpha_doubleexcitations=[]
        for i in range(neh):
            for j in range(i+1,neh):
                for k in range(neh,self.num_bas):
                    for l in range(k+1,self.num_bas):
                        newstring=groundstring[:i]+"0"+groundstring[i+1:j]+"0"+groundstring[j+1:k]+"1"+groundstring[k+1:l]+"1"+groundstring[l+1:]#Take the ith & jth 1 and move it to positions k and l
                        alpha_doubleexcitations.append(newstring)
        GS=[[groundstring,groundstring]]
        singles_alpha=[[alpha,groundstring] for alpha in alpha_singleexcitations] #All single excitations within alpha
        singles_beta=[[groundstring,alpha] for alpha in alpha_singleexcitations]
        doubles_alpha=[[alpha,groundstring] for alpha in alpha_doubleexcitations]
        doubles_beta=[[groundstring,alpha] for alpha in alpha_doubleexcitations]
        doubles_alphabeta=[[alpha,beta] for alpha in alpha_singleexcitations for beta in alpha_singleexcitations]
        allstates=GS+singles_alpha+singles_beta+doubles_alpha+doubles_beta+doubles_alphabeta
        return allstates
    def index_creator(self):
        all_indices=[]
        for state in self.states:
            alphas_occ=[i for i in range(len(state[0])) if int(state[0][i])==1]
            betas_occ=[i for i in range(len(state[1])) if int(state[1][i])==1]
            all_indices.append([alphas_occ,betas_occ])
        return all_indices
    def make_T(self):
        self.T=np.diag(self.diagonal_energy()+self.mol.energy_nuc())
        self.T+=self.offdiagonal_energy()
    def solve_T(self):
        if self.T is None:
            self.make_T()
        self.energy,self.eigenvector=np.linalg.eigh(self.T)
        return self.energy[0],self.eigenvector[:,0]
    def offdiagonal_energy(self):
        states=self.states
        indices=self.indices
        states_fallen=self.states_fallen
        onebody=self.onebody
        twobody=self.twobody
        num_bas=self.num_bas
        T=np.zeros((len(states),len(states)))
        for i in range(len(states)):
            for j in range(i+1,len(states)):
                diffalpha,diffbeta=calc_diff_bitstring(states_fallen[i],states_fallen[j])
                state_difference=diffalpha+diffbeta
                if (state_difference<=4): #Only then, nonzero contribution
                    e1=0
                    e2=0
                    alpha_left=indices[i][0]
                    beta_left=indices[i][1]
                    if diffalpha==2 and diffbeta==0: #A single "excitation" from one alpha state to another alpha state
                        m=[x for x in range(num_bas) if states[i][0][x]=="1" and states[j][0][x]=="0"][0]
                        p=[x for x in range(num_bas) if states[i][0][x]=="0" and states[j][0][x]=="1"][0]
                        al=indices[i][0].copy()
                        ar=indices[j][0].copy()
                        al[np.where(np.array(al)==m)[0][0]]=p
                        parity_here=parity(np.argsort(al))*parity(np.argsort(ar))
                        e1=onebody[m,p]
                        for n in alpha_left:
                            e2+=twobody[m,p,n,n]
                            e2-=twobody[m,n,n,p]
                        for n in beta_left:
                            e2+=twobody[m,p,n,n]
                    elif diffbeta==2 and diffalpha==0:
                        m=[x for x in range(num_bas) if states[i][1][x]=="1" and states[j][1][x]=="0"][0]
                        p=[x for x in range(num_bas) if states[i][1][x]=="0" and states[j][1][x]=="1"][0]
                        bl=indices[i][1].copy()
                        br=indices[j][1].copy()
                        bl[np.where(np.array(bl)==m)[0][0]]=p
                        parity_here=parity(np.argsort(bl))*parity(np.argsort(br))
                        e1=onebody[m,p]
                        for n in alpha_left:
                            e2+=twobody[m,p,n,n]
                        for n in beta_left:
                            e2+=twobody[m,p,n,n]
                            e2-=twobody[m,n,n,p]
                    elif diffalpha==4:
                        al=indices[i][0].copy()
                        ar=indices[j][0].copy()
                        m,n=[x for x in range(num_bas) if states[i][0][x]=="1" and states[j][0][x]=="0"]
                        p,q=[x for x in range(num_bas) if states[i][0][x]=="0" and states[j][0][x]=="1"]
                        al[np.where(np.array(al)==m)[0][0]]=p
                        al[np.where(np.array(al)==n)[0][0]]=q
                        parity_here=parity(np.argsort(al))*parity(np.argsort(ar))
                        e2=twobody[m,p,n,q]-twobody[m,q,n,p]
                    elif diffbeta==4:
                        bl=indices[i][1].copy()
                        br=indices[j][1].copy()
                        m,n=[x for x in range(num_bas) if states[i][1][x]=="1" and states[j][1][x]=="0"]
                        p,q=[x for x in range(num_bas) if states[i][1][x]=="0" and states[j][1][x]=="1"]
                        bl[np.where(np.array(bl)==m)[0][0]]=p
                        bl[np.where(np.array(bl)==n)[0][0]]=q
                        parity_here=parity(np.argsort(bl))*parity(np.argsort(br))
                        e2=twobody[m,p,n,q]-twobody[m,q,n,p]
                    elif diffalpha==2 and diffbeta==2:
                        al=indices[i][0].copy()
                        ar=indices[j][0].copy()
                        bl=indices[i][1].copy()
                        br=indices[j][1].copy()
                        m=[x for x in range(num_bas) if states[i][0][x]=="1" and states[j][0][x]=="0"][0]
                        n=[x for x in range(num_bas) if states[i][1][x]=="1" and states[j][1][x]=="0"][0]
                        p=[x for x in range(num_bas) if states[i][0][x]=="0" and states[j][0][x]=="1"][0]
                        q=[x for x in range(num_bas) if states[i][1][x]=="0" and states[j][1][x]=="1"][0]
                        al[np.where(np.array(al)==m)[0][0]]=p
                        bl[np.where(np.array(bl)==n)[0][0]]=q
                        parity_here=parity(np.argsort(bl))*parity(np.argsort(br))*parity(np.argsort(al))*parity(np.argsort(ar))
                        e2=twobody[m,p,n,q]
                    T[i,j]=T[j,i]=parity_here*(e1+e2)
        return T
    def diagonal_energy(self):
        num_states=self.num_states
        indices=self.indices
        onebody=self.onebody
        twobody=self.twobody
        diagonal_matrix=np.empty(num_states)
        for i in range(num_states):
            alpha=indices[i][0]
            beta=indices[i][1]
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

            diagonal_matrix[i]=onebody_alpha+onebody_beta+energy2

        return diagonal_matrix
def make_mol(molecule,x,basis="6-31G"):
    mol=gto.Mole()
    mol.atom=molecule(x)
    mol.basis = basis
    mol.unit= "Bohr"
    mol.build()
    return mol
def create_reference_determinant(mol):
    mf=scf.RHF(mol)
    mf.kernel()
    mo_coeff_converted=localize_cholesky(mol,mf.mo_coeff,mf.mo_occ)
    return mo_coeff_converted
def evc(T,states,nonzeros=None):
    T_subspace=np.zeros((len(states),len(states)))
    S_subspace=np.zeros_like(T_subspace)
    if nonzeros is None:
        nonzeros=np.arange(len(states[0]))
    for i in range(len(states)):
        for j in range(i,len(states)):
            T_subspace[j,i]=T_subspace[i,j]=states[i][nonzeros]@T@states[j][nonzeros]
            S_subspace[i,j]=S_subspace[j,i]=states[i][nonzeros]@states[j][nonzeros]
    e,vec=generalized_eigenvector(T_subspace,S_subspace)
    return e,vec


molecule=lambda x: "H 0 0 0; F 0 0 %f"%x
#molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
basis="6-31G"
molecule_name="H F"
basischange=False
x_sol=np.linspace(1.2,4.5,34)
#x_sol=np.array([1.9,2.5])
ref_x_index=[0,4,8,12,16,33]
reference_position=13
#ref_x_index=[0]
ref_x=[x_sol[ref_x_index]][0]
print(ref_x)
energy_refx=[]
energies_EVC=np.zeros((len(x_sol),len(ref_x_index)))
energies_ref=np.zeros(len(x_sol))
mol=make_mol(molecule,x_sol[reference_position],basis)
reference_determinant=create_reference_determinant(mol)
all_determinants=[]
reference_solutions=[]
excitation_operators=[]
mo_coeffs=[]
#Step 1: Find the Slater-determinants

for index, x in enumerate(x_sol):
    mol=make_mol(molecule,x,basis)
    mf=scf.RHF(mol)
    mf.kernel()
    all_determinants.append(localize_cholesky(mol,mf.mo_coeff,mf.mo_occ))

#Step 2: Minimize in relation to some reference
for i, x in enumerate(x_sol):
    #all_determinants[i]=localize_procrustes(mol,all_determinants[i],mf.mo_occ,reference_determinant) #Minimize to reference
    pass
#Step 3: Get excitation coefficients at reference geometries
for i in ref_x_index:
    x=x_sol[i]
    print(x)
    mol=make_mol(molecule,x,basis)
    #solver=RHF_CISDsolver(mol,mo_coeff=all_determinants[i])
    solver=RHF_CISDsolver(mol,mo_coeff=reference_determinant,basischange=True)
    energy,sol=solver.solve_T()
    energy_refx.append(energy)
    reference_solutions.append(sol)
"""
allstates=np.zeros(len(reference_solutions[0]))
nonzeros=[]
for state in reference_solutions:
    allstates+=np.abs(state)
counter=0
allstates/=len(reference_solutions)
for index in range(len(allstates)):
    destroyer=10**(-3.5)
    if allstates[index]<destroyer:# and allstates[index]>1e-13:
        counter+=1
        for state in reference_solutions:
            state[index]=0
    else:
        nonzeros.append(index)
for state in reference_solutions:
    norm=state.T@state
    print(norm)
    state=state/norm #Renormalize
print(counter,len(allstates[allstates>1e-13]),counter/len(allstates[allstates>1e-13]))
"""
nonzeros=None

#Step 4: Solve problem for each x.
for i,x in enumerate(x_sol):
    print(i)
    mol=make_mol(molecule,x,basis)
    #solver=RHF_CISDsolver(mol,mo_coeff=all_determinants[i],nonzeros=nonzeros)
    solver=RHF_CISDsolver(mol,mo_coeff=reference_determinant,basischange=True)
    solver.make_T()
    e_corr,sol0=solver.solve_T()
    excitation_operators.append(sol0*np.sign(sol0[0]))
    mo_coeffs.append(solver.mo_coeff)
    for j in range(1,len(reference_solutions)+1):
        e,sol=evc(solver.T,reference_solutions[:j],nonzeros)
        energies_EVC[i,j-1]=e
    energies_ref[i]=e_corr
diff_norms=[]
diff_coeffs=[]
for i in range(0,len(mo_coeffs)):
    diff_coeffs.append(np.linalg.norm((mo_coeffs[i]-reference_determinant)))
    diff_exc1=np.linalg.norm((excitation_operators[i]-excitation_operators[reference_position]))
    diff_exc2=np.linalg.norm((excitation_operators[i]+excitation_operators[reference_position]))
    diff_norms.append(np.min([diff_exc1,diff_exc2]))

fig,ax=plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False)
fig.suptitle("%s (%s)"%(molecule_name,basis))
for i in range(len(ref_x_index)):
    ax[0].plot(x_sol,energies_EVC[:,i],label="EVC (%d)"%i)
ax[0].plot(x_sol,energies_ref,label="CISD")
ax[0].plot(ref_x,energy_refx,"*",label="Sample points")
ax[0].legend(loc='upper left',handletextpad=0.1)

ax[0].set_xlabel("Internuclear distance (Bohr)")
ax[0].set_ylabel("Energy")
ax[1].plot(x_sol,diff_norms,label="exc. coefficients")
ax[1].set_title("Norm of difference to reference")
ax[1].plot(x_sol,diff_coeffs,label="MO coefficients")
ax[1].set_xlabel("Internuclear distance (Bohr)")
ax[1].set_ylabel("Norm")
ax[1].legend()
plt.tight_layout()
if reference_determinant is None:
    filename="%s_%s_NONE.pdf"%(molecule_name,basis)
else:
    filename="%s_%s.pdf"%(molecule_name,basis)
plt.savefig(filename)
plt.show()
for i,mo_coeff in enumerate(mo_coeffs):
    print(x_sol[i])
    print(mo_coeff)
