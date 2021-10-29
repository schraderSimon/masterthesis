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
from scipy.optimize import minimize
np.set_printoptions(linewidth=200,precision=5,suppress=True)
def similarize_1(MO_tochange,MO_ref,f,dev):
    num_bas=len(MO_tochange[0,:])
    for i in range(num_bas):
        for j in range(i+1,num_bas):
            x1=MO_ref[:,i]
            x2=MO_ref[:,j]
            y1=MO_tochange[:,i]
            y2=MO_tochange[:,j]
            before=f(0,y1,y2,x1,x2)
            alpha=minimize(f,0,args=(y1,y2,x1,x2)).x[0]#,jac=dev,method="BFGS").x[0]#,jac=derivative_of_function).x[0]
            after=f(alpha,y1,y2,x1,x2)
            sa=np.sin(alpha)
            ca=np.cos(alpha)
            y1_new=ca*MO_tochange[:,i]-sa*MO_tochange[:,j]
            y2_new=sa*MO_tochange[:,i]+ca*MO_tochange[:,j]
            if before<after:
                print("You made it worse")

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
            before=f(0,y1,y2,x1,x2)
            alpha=minimize(f,0,args=(y1,y2,x1,x2)).x[0]#,jac=dev,method="BFGS").x[0]#,jac=derivative_of_function).x[0]
            after=f(alpha,y1,y2,x1,x2)
            sa=np.sin(alpha)
            ca=np.cos(alpha)
            y1_new=(ca*y1+sa*y2-x1)
            y2_new=(sa*y1-ca*y2-x2)
            after=f(0,y1_new,y2_new,x1,x2)
            if before<after:
                print(alpha)
                print(f(0,y1,y2,x1,x2,printerino=True))
                print(f(0,y1_new,y2_new,x1,x2,printerino=True))
                print(f(alpha,y1,y2,x1,x2,printerino=True))
                print("You made it worse")
                sys.exit(1)
            MO_tochange[:,i]=y1_new
            MO_tochange[:,j]=y2_new
    return MO_tochange

def orbital_dissimilarity(alpha,y1,y2,x1,x2,printerino=False):
    ca=np.cos(alpha)
    sa=np.sin(alpha)
    first=np.sum((ca*y1+sa*y2-x1)**2)
    second=np.sum((sa*y1-ca*y2-x2)**2)
    if printerino:
        print(first)
        print(second)
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
    first=np.sum((ca*y1-sa*y2-x1)**2)
    second=np.sum((sa*y1+ca*y2-x2)**2)
    return first+second
def orbital_dissimilarity_dev_1(alpha,y1,y2,x1,x2):
    ca=np.cos(alpha)
    sa=np.sin(alpha)
    first=np.sum((ca*y1-sa*y2-x1)*(-sa*y1-ca*y2))
    second=np.sum((sa*y1+ca*y2-x2)*((ca*y1-sa*y2)))
    return first+second


def localize_mocoeff(mol,mo_coeff,mo_occ,previous_mo_coeff=None):

    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ>0])
    mo=swappistan(mo)
    if previous_mo_coeff is not None:
        premo=previous_mo_coeff[:,mo_occ>0]
        #mo=similarize(mo,premo,f=orbital_dissimilarity,dev=orbital_dissimilarity_dev)
        mo=similarize_1(mo,premo,f=orbital_dissimilarity_1,dev=orbital_dissimilarity_dev_1)
        for i in range(len(mo[0,:])):
            if np.sum(np.abs(mo[:,i]-premo[:,i])**2)>np.sum(np.abs(mo[:,i]+premo[:,i])**2):
                mo[:,i]=-mo[:,i]
    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo_unocc=cholesky_coefficientmatrix(mo_coeff[:,mo_occ<=0])
    mo_unocc=swappistan(mo_unocc)

    if previous_mo_coeff is not None:
        premo=previous_mo_coeff[:,mo_occ<=0]
        print("Norm before: %f"%np.sum(np.abs((mo_unocc-premo))**2))
        mo_unocc=similarize(mo_unocc,premo,f=orbital_dissimilarity,dev=orbital_dissimilarity_dev)
        #mo_unocc=similarize_1(mo_unocc,premo,f=orbital_dissimilarity_1,dev=orbital_dissimilarity_dev_1)
        for i in range(len(mo_unocc[0,:])):
            if np.sum(np.abs(mo_unocc[:,i]-premo[:,i])**2)>np.sum(np.abs(mo_unocc[:,i]+premo[:,i])**2):
                mo_unocc[:,i]=-mo_unocc[:,i]
        print("Norm after: %f"%np.sum(np.abs((mo_unocc-premo))**2))
    mo_coeff[:,mo_occ<=0]=np.array(mo_unocc)

    return mo_coeff

"""
def localize_mocoeff(mol,mo_coeff,mo_occ,previous_mo_coeff=None):

    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ>0])
    mo=swappistan(mo)
    if previous_mo_coeff is not None:
        previous_mo=previous_mo_coeff[:,mo_occ>0]
        diff=mo-previous_mo
        #Do the magic and swap columns and rows
        col_norm=np.sum(np.abs(diff),axis=0)
        colnorm_descendingargs=np.argsort(col_norm)[::-1]
        colnorm_sorted=col_norm[colnorm_descendingargs]
        if colnorm_sorted[0]>0.9:
            mo=swap_cols(mo,colnorm_descendingargs[0],colnorm_descendingargs[1])

    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ<=0])
    mo=swappistan(mo)
    if previous_mo_coeff is not None:
        previous_mo=previous_mo_coeff[:,mo_occ<=0]
        diff=mo-previous_mo
        #Do the magic and swap columns and rows
        col_norm=np.sum(np.abs(diff),axis=0)
        #print(np.abs(diff))
        #print(col_norm)
        colnorm_descendingargs=np.argsort(col_norm)[::-1]
        colnorm_sorted=col_norm[colnorm_descendingargs]
        if colnorm_sorted[0]>1:
            mo=swap_cols(mo,colnorm_descendingargs[0],colnorm_descendingargs[1])
    mo_coeff[:,mo_occ<=0]=np.array(mo)

    return mo_coeff
"""
def calc_diff_bitstring(a,b):
    difference_alpha=int(a[:len(a)//2],2)^int(b[:len(b)//2],2)
    difference_beta=int(a[len(a)//2:],2)^int(b[len(b)//2:],2)
    ndiffa=bin(difference_alpha).count("1") #How many differ
    ndiffb=bin(difference_beta).count("1") #How many differ
    return ndiffa,ndiffb

class RHF_CISDsolver():
    def __init__(self,mol,mo_coeff=None,previous_mo_coeff=None):
        self.mol=mol
        self.overlap=mol.intor("int1e_ovlp")
        if mo_coeff is None:
            print("Finding a new SL-det")
            mf=scf.RHF(mol)
            mf.kernel()
            self.mo_coeff=mf.mo_coeff
            self.mo_coeff=localize_mocoeff(mol,mf.mo_coeff,mf.mo_occ,previous_mo_coeff)
        else:
            self.mo_coeff=self.basischange(mo_coeff,self.overlap)
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
        self.data_creator()
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
        self.states=states
        states_fallen=[state[0]+state[1] for state in states]
        indices=np.array(self.index_creator())

        self.num_states=len(states)
        self.states_fallen=states_fallen
        self.indices=indices
    def index_creator(self):
        all_indices=[]
        for state in self.states:
            alphas_occ=[i for i in range(len(state[0])) if int(state[0][i])==1]
            betas_occ=[i for i in range(len(state[1])) if int(state[1][i])==1]
            all_indices.append([alphas_occ,betas_occ])
        return all_indices
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
    mo_coeff_converted=localize_mocoeff(mol,mf.mo_coeff,mf.mo_occ)
    return mo_coeff_converted
def evc(T,states):
    T_subspace=np.zeros((len(states),len(states)))
    S_subspace=np.zeros_like(T_subspace)

    for i in range(len(states)):
        for j in range(i,len(states)):
            T_subspace[j,i]=T_subspace[i,j]=states[i]@T@states[j]
            S_subspace[i,j]=S_subspace[j,i]=states[i]@states[j]
    e,vec=generalized_eigenvector(T_subspace,S_subspace)
    return e,vec


molecule=lambda x: "H 0 0 0; Li 0 0 %f"%x
#molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))

basis="6-31G"
molecule_name="LIH"
mol=make_mol(molecule,2,basis)
ref_x=[2,2.5,3]
energy_refx=[]
x_sol=np.linspace(2,5,31)
energies_EVC=np.zeros(len(x_sol))
energies_ref=np.zeros(len(x_sol))
reference_determinant=create_reference_determinant(mol)
reference_solutions=[]
for x in ref_x:
    print(x)
    mol=make_mol(molecule,x,basis)
    solver=RHF_CISDsolver(mol,previous_mo_coeff=reference_determinant)
    energy,sol=solver.solve_T()
    energy_refx.append(energy)
    reference_solutions.append(sol)
allstates=np.zeros(len(reference_solutions[0]))
for state in reference_solutions:
    allstates+=np.abs(state)
counter=0
allstates/=len(reference_solutions)
for index in range(len(allstates)):
    destroyer=1e-10
    if allstates[index]<destroyer and allstates[index]>1e-13:
        counter+=1
        for state in reference_solutions:
            state[index]=0
for state in reference_solutions:
    norm=state.T@state
    print(norm)
    state=state/norm #Renormalize
print(counter,len(reference_solutions[0]),counter/len(reference_solutions[0]))
excitation_operators=[]
mo_coeffs=[]


mol=make_mol(molecule,x_sol[0],basis)
solver=RHF_CISDsolver(mol,previous_mo_coeff=reference_determinant)
#solver=RHF_CISDsolver(mol,reference_determinant)
solver.make_T()
e_corr,sol0=solver.solve_T()
excitation_operators.append(sol0*np.sign(np.max(sol0)))
mo_coeffs.append(solver.mo_coeff)
e,sol=evc(solver.T,reference_solutions)
energies_EVC[0]=e
energies_ref[0]=e_corr
for i,x in enumerate(x_sol):
    if i==0:
        continue
    print(i)
    mol=make_mol(molecule,x,basis)
    #solver=RHF_CISDsolver(mol,reference_determinant)
    solver=RHF_CISDsolver(mol,previous_mo_coeff=reference_determinant)
    solver.make_T()
    e_corr,sol0=solver.solve_T()
    excitation_operators.append(sol0*np.sign(sol0[0]))
    mo_coeffs.append(solver.mo_coeff)
    e,sol=evc(solver.T,reference_solutions)
    energies_EVC[i]=e
    energies_ref[i]=e_corr
diff_norms=[0]
diff_coeffs=[0]
for i in range(1,len(mo_coeffs)):
    diff_coeffs.append(np.sum(np.abs(mo_coeffs[i]-mo_coeffs[i-1])))
    diff_exc1=np.sqrt(np.sum(np.abs(excitation_operators[i]-excitation_operators[i-1])**2))
    diff_exc2=np.sqrt(np.sum(np.abs(excitation_operators[i]+excitation_operators[i-1])**2))
    diff_norms.append(np.min([diff_exc1,diff_exc2]))
fig,ax=plt.subplots(nrows=3,ncols=1,sharex=True)
fig.suptitle("%s (%s)"%(molecule_name,basis))
ax[0].plot(x_sol,energies_EVC,label="EVC")
ax[0].plot(x_sol,energies_ref,label="CISD")
ax[0].plot(ref_x,energy_refx,"*",label="Sample points")

ax[0].legend(loc='upper left')

plt.xlabel("Internuclear distance (Bohr)")
ax[0].set_ylabel("Energy")
ax[1].plot(x_sol,diff_norms,label="Norm exc. coefficients (absolute value)")
ax[1].set_title("Norm change in (correct) excitation operators")
ax[2].plot(x_sol,diff_coeffs,label="Norm MO coefficients")
ax[2].set_title("Norm change in MO coefficients")
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
