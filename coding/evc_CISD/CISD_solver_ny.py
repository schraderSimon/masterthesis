import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf,ao2mo,lo,ci,fci
import scipy
import sys
from numba import jit
import scipy
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
    #return matrix
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


#@jit
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
#@jit
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

def calc_diff_bitstring(a,b):
    difference_alpha=int(a[:len(a)//2],2)^int(b[:len(b)//2],2)
    difference_beta=int(a[len(a)//2:],2)^int(b[len(b)//2:],2)
    ndiffa=bin(difference_alpha).count("1") #How many differ
    ndiffb=bin(difference_beta).count("1") #How many differ
    return ndiffa,ndiffb
def state_creator(N_elec_half,N_basis_spatial):
    groundstring="1"*N_elec_half+"0"*(N_basis_spatial-N_elec_half)
    alpha_singleexcitations=[]
    for i in range(N_elec_half):
        for j in range(N_elec_half,N_basis_spatial):
            newstring=groundstring[:i]+"0"+groundstring[i+1:j]+"1"+groundstring[j+1:]
            alpha_singleexcitations.append(newstring)
    alpha_doubleexcitations=[]
    for i in range(N_elec_half):
        for j in range(i+1,N_elec_half):
            for k in range(N_elec_half,N_basis_spatial):
                for l in range(k+1,N_basis_spatial):
                    newstring=groundstring[:i]+"0"+groundstring[i+1:j]+"0"+groundstring[j+1:k]+"1"+groundstring[k+1:l]+"1"+groundstring[l+1:]
                    alpha_doubleexcitations.append(newstring)
    GS=[[groundstring,groundstring]]
    singles_alpha=[[alpha,groundstring] for alpha in alpha_singleexcitations]
    singles_beta=[[groundstring,alpha] for alpha in alpha_singleexcitations]
    doubles_alpha=[[alpha,groundstring] for alpha in alpha_doubleexcitations]
    doubles_beta=[[groundstring,alpha] for alpha in alpha_doubleexcitations]
    doubles_alphabeta=[[alpha,beta] for alpha in alpha_singleexcitations for beta in alpha_singleexcitations]
    allstates=GS+singles_alpha+singles_beta+doubles_alpha+doubles_beta+doubles_alphabeta
    return allstates
def index_creator(states,N_elec_half,N_basis_spatial):
    all_indices=[]
    for state in states:
        alphas_occ=[i for i in range(len(state[0])) if int(state[0][i])==1]
        betas_occ=[i for i in range(len(state[1])) if int(state[1][i])==1]
        all_indices.append([alphas_occ,betas_occ])
    return all_indices
def basischange_alt(C_old,overlap_AOs_newnew):
    S=np.einsum("mi,vj,mv->ij",C_old,C_old,overlap_AOs_newnew)
    S_eig,S_U=np.linalg.eigh(S)
    S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
    C_new=np.einsum("ij,mj->mi",S_powerminusonehalf,C_old)
    return C_new
def offdiagonal_energy(states,indices,states_fallen,onebody,twobody):
    T=np.zeros((len(states),len(states)))
    for state in states:
        print(state)
    print("Start loop")
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
                    print("%d->%d"%(m,p))
                    print(states_fallen[i])
                    print(states_fallen[j])
                    e1=onebody[m,p]
                    for n in alpha_left:
                        e2+=twobody[m,p,n,n]
                        e2-=twobody[m,n,n,p]
                    for n in beta_left:
                        e2+=twobody[m,p,n,n]
                elif diffbeta==2 and diffalpha==0:
                    m=[x for x in range(num_bas) if states[i][1][x]=="1" and states[j][1][x]=="0"][0]
                    p=[x for x in range(num_bas) if states[i][1][x]=="0" and states[j][1][x]=="1"][0]
                    e1=onebody[m,p]
                    for n in alpha_left:
                        e2+=twobody[m,p,n,n]
                    for n in beta_left:
                        e2+=twobody[m,p,n,n]
                        e2-=twobody[m,n,n,p]
                elif diffalpha==4:
                    m,n=[x for x in range(num_bas) if states[i][0][x]=="1" and states[j][0][x]=="0"]
                    p,q=[x for x in range(num_bas) if states[i][0][x]=="0" and states[j][0][x]=="1"]
                    e2=twobody[m,p,n,q]-twobody[m,q,n,p]
                    #print("%d->%d %d->%d"%(m,p,n,q))
                    #print(states_fallen[i])
                    #print(states_fallen[j])
                elif diffbeta==4:
                    m,n=[x for x in range(num_bas) if states[i][1][x]=="1" and states[j][1][x]=="0"]
                    p,q=[x for x in range(num_bas) if states[i][1][x]=="0" and states[j][1][x]=="1"]
                    e2=twobody[m,p,n,q]-twobody[m,q,n,p]
                elif diffalpha==2 and diffbeta==2:
                    m=[x for x in range(num_bas) if states[i][0][x]=="1" and states[j][0][x]=="0"][0]
                    n=[x for x in range(num_bas) if states[i][1][x]=="1" and states[j][1][x]=="0"][0]
                    p=[x for x in range(num_bas) if states[i][0][x]=="0" and states[j][0][x]=="1"][0]
                    q=[x for x in range(num_bas) if states[i][1][x]=="0" and states[j][1][x]=="1"][0]
                    #print("%d->%d %d->%d"%(m,p,n+num_bas,q+num_bas))
                    #print(states_fallen[i])
                    #print(states_fallen[j])
                    e2=twobody[m,p,n,q]
                T[i,j]=T[j,i]=e1+e2
    print("End loop")
    return T
#@jit
def diagonal_energy(indices,onebody,twobody,num_states):
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
                energy2+=twobody[b,b,a,a]
        for a in beta:
            for b in beta:
                energy2+=twobody[a,a,b,b]
                energy2-=twobody[a,b,b,a]
        energy2*=0.5
        diagonal_matrix[i]=onebody_alpha+onebody_beta+energy2
    return diagonal_matrix
def make_mol(molecule,x):
    basis="6-31G"
    molname="HF"
    mol=gto.Mole()
    mol.atom=molecule(x)
    mol.basis = basis
    mol.unit= "Bohr"
    mol.build()
    return mol
def data_creator(neh,num_bas):
    states=state_creator(neh,num_bas)
    states=state_creator(neh,num_bas)
    states_fallen=[state[0]+state[1] for state in states]
    indices=np.array(index_creator(states,neh,num_bas))
    return states,states_fallen,indices
def energy_bitch(default,T):
    return default.T@T@default

molecule=lambda x: "H 0 0 0; Li 0 0 %f"%x
mol=make_mol(molecule,1.5)
mf=scf.RHF(mol)
mf.kernel()
cisd=ci.CISD(mf).run()

overlap_matrix_ref=mol.intor("int1e_ovlp")
#print("Pre-change")
#print(localize_mocoeff(mol,mf.mo_coeff,mf.mo_occ))
basis_mo_coeff=basischange_alt(localize_mocoeff(mol,mf.mo_coeff,mf.mo_occ),overlap_matrix_ref)
#basis_mo_coeff=basischange_alt(mf.mo_coeff,overlap_matrix_ref)
#print("Post-change")
#print(basis_mo_coeff)
occ_bas=basis_mo_coeff[:,:mol.nelectron//2]
unocc_bas=basis_mo_coeff[:,mol.nelectron//2:]
bitcherino=basischange_alt(basis_mo_coeff,overlap_matrix_ref)

try:

    assert(np.all(np.abs(basis_mo_coeff-bitcherino)<1e-10)), "They are somehow different???"
except:
    print("They are different")
    print("Occupied density matrix difference")
    occ=mf.mo_coeff[:,:mol.nelectron//2]
    unocc=mf.mo_coeff[:,mol.nelectron//2:]
    occ_bitch=bitcherino[:,:mol.nelectron//2]
    unocc_bitch=bitcherino[:,mol.nelectron//2:]
    print(occ@occ.T-occ_bitch@occ_bitch.T)
    print("Unocc difference")
    print(unocc@unocc.T-unocc_bitch@unocc_bitch.T)

#sys.exit(1)
onebody_matrix_ref=np.einsum("ki,lj,kl->ij",basis_mo_coeff,basis_mo_coeff,mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
twobody_matrix_ref=ao2mo.get_mo_eri(mol.intor('int2e',aosym="s1"),(basis_mo_coeff,basis_mo_coeff,basis_mo_coeff,basis_mo_coeff),aosym="s1")
states,states_fallen,indices=data_creator(int(mol.nelectron//2),(basis_mo_coeff.shape[0]))
num_bas=(basis_mo_coeff.shape[0])
diagonal_matrix_ref=diagonal_energy(indices,onebody_matrix_ref,twobody_matrix_ref,len(indices))+mol.energy_nuc()

offdiagonal_ref=offdiagonal_energy(states,indices,states_fallen,onebody_matrix_ref,twobody_matrix_ref)
T=offdiagonal_ref+np.diag(diagonal_matrix_ref)
eigenvalues_ref,eigenvectors_ref=np.linalg.eigh(T)
reference_excitations=eigenvectors_ref[:,0]
assert(np.all(np.abs(eigenvectors_ref[:,0]*eigenvalues_ref[0]-T@eigenvectors_ref[:,0])<1e-10))

xvals=np.linspace(3,6,10)
energies=np.zeros_like(xvals)
energies_default=np.zeros_like(xvals)
energies_pyscf=np.zeros_like(xvals)

for indexerino,x in enumerate(xvals):
    mol=make_mol(molecule,x)
    overlap_matrix=mol.intor("int1e_ovlp")
    #mo_coeff=basischange_alt(basis_mo_coeff,overlap_matrix)
    mf=scf.RHF(mol)
    mf.kernel()
    #mo_coeff=localize_mocoeff(mol,mf.mo_coeff,mf.mo_occ)
    mo_coeff=mf.mo_coeff #Canonical orbitals
    cisd=ci.CISD(mf).run()
    #fcisolver=fci.FCI(mf)
    energies_pyscf[indexerino]=cisd.e_tot
    onebody_matrix=np.einsum("ki,lj,kl->ij",mo_coeff,mo_coeff,mol.intor("int1e_kin")+mol.intor("int1e_nuc"))
    twobody_matrix=ao2mo.get_mo_eri(mol.intor('int2e',aosym="s1"),(mo_coeff,mo_coeff,mo_coeff,mo_coeff),aosym="s1")
    states,states_fallen,indices=data_creator(int(mol.nelectron/2),(mo_coeff.shape[0]))
    num_bas=(mo_coeff.shape[0])
    diagonal_matrix=diagonal_energy(indices,onebody_matrix,twobody_matrix,len(indices))+mol.energy_nuc()
    offdiagonals=offdiagonal_energy(states,indices,states_fallen,onebody_matrix,twobody_matrix)
    T=offdiagonals+np.diag(diagonal_matrix)
    eigenvalues,eigenvectors=np.linalg.eigh(T)
    energies[indexerino]=eigenvalues[0]
    energies_default[indexerino]=energy_bitch(reference_excitations,T)
    print("E_EVC: %.5f"%energy_bitch(reference_excitations,T))
    print("Own CISD: %.5f"%eigenvalues[0])
    print("pyscf cisd: %.5f"%cisd.e_tot)
    #print("pyscf fci: %.5f"%fcisolver.kernel()[0])
    print("Diagonal elements")
    print(T)
plt.plot(xvals,energies,label="Crap-referanse")
plt.plot(xvals,energies_default,label="Default")
plt.plot(xvals,energies_pyscf,label="pyscf")

plt.legend()
plt.show()
