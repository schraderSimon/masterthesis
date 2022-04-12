import numpy as np
from pyscf import gto, scf, fci,cc,ao2mo, mp, mcscf
import pyscf
import sys
import scipy
from matrix_operations import *
from helper_functions import *
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
np.set_printoptions(linewidth=200,precision=4,suppress=True)
import matplotlib
class eigvecsolver_RHF():
    def __init__(self,sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False,type="transform",coeff_matrices=None):
        """Initiate the solver.
        It Creates the HF coefficient matrices for the sample points for a given molecule.

        Input:
        sample_points (array) - the points at which to evaluate the WF
        basis_type - atomic basis
        molecule - A function f(x), returning the molecule as function of x
        """
        if coeff_matrices is None:
            self.HF_coefficients=[] #The Hartree Fock coefficients solved at the sample points
        else:
            self.HF_coefficients=coeff_matrices
        self.molecule=molecule
        self.basis_type=basis_type
        self.spin=spin
        self.unit=unit
        self.symmetry=symmetry
        self.charge=charge
        self.sample_points=sample_points
        self.type=type
        self.solve_HF()
    def solve_HF(self):
        """
        Create coefficient matrices for each sample point
        """
        HF_coefficients=self.HF_coefficients
        mol=self.build_molecule(1)
        if len(HF_coefficients)==0:
            for x in self.sample_points:
                mol=self.build_molecule(x)
                mf=mol.RHF().run(verbose=2,conv_tol=1e-8)
                expansion_coefficients_mol= mf.mo_coeff[:, mf.mo_occ >=-1] #Use all
                HF_coefficients.append(expansion_coefficients_mol)
        self.HF_coefficients=HF_coefficients
        self.nMO_h=HF_coefficients[0].shape[0] #Number of molecular orbitals divided by two (for each spin)
        self.num_bas=self.nMO_h
        self.n_unocc=self.num_bas-self.ne_h
    def build_molecule(self,x):
        """Create a molecule object with parameter x"""

        mol=gto.Mole()
        mol.atom="""%s"""%(self.molecule(x))
        mol.charge=self.charge
        mol.spin=self.spin
        mol.unit=self.unit
        mol.symmetry=self.symmetry
        mol.basis=self.basis_type
        mol.build()
        self.number_electronshalf=int(mol.nelectron/2)
        self.ne_h=self.number_electronshalf
        return mol
    def calculate_energies(self,xc_array):
        """Calculates the molecule's energy"""
        energy_array=np.zeros(len(xc_array))
        eigvec_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.HF_coefficients)):
                new_HF_coefficients.append(self.change_coefficients(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf])
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            try:
                eigval,eigvec=generalized_eigenvector(T,S,threshold=1e-12)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval
            eigvec_array.append(eigvec)
        self.eigvecs=eigvec_array
        return energy_array,eigvec_array
    def change_coefficients(self,C_old, S_new):
        if self.type=="transform":
            returnmat=basischange(C_old,S_new,self.number_electronshalf)
        elif self.type=="procrustes":
            C_new=np.real(fractional_matrix_power(S_new,-0.5))
            #weights=np.zeros(len(C_new))
            #weights[:self.number_electronshalf]=1
            returnmat=localize_procrustes(None,C_new,None,C_old,mix_states=True,nelec=self.number_electronshalf,weights=None)
        return returnmat
    def getdeterminant_matrix(self,AO_overlap,HF_coefficients_left,HF_coefficients_right):
        determinant_matrix=contract("ab,ai,bj->ij",AO_overlap,HF_coefficients_left,HF_coefficients_right)
        return determinant_matrix

    def biorthogonalize(self,C_w,C_x,AO_overlap):
        """Biorthogonalizes the problem
        Input: The left and right Slater determinants (in forms of coefficient matrixes)
        Returns: The diagonal matrix (as array) D, the new basises for C_w and C_x.
        """
        S=self.getdeterminant_matrix(AO_overlap,C_w,C_x);
        Linv,D,Uinv=LDU_decomp(S,overwrite_a=True)
        C_w=C_w@Linv
        C_x=C_x@Uinv.T
        return D,C_w,C_x
    def getoverlap(self,determinant_matrix):
        overlap=np.linalg.det(determinant_matrix)**2
        return overlap
    def calculate_overlap_to_HF(self,xc_array):
        energies,eigvecs=self.calculate_energies(xc_array) #get energies and eigenvalues
        overlap_to_HF=np.zeros(len(energies))
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)

            "Calculate RHF solution"
            mol_xc=self.build_molecule(xc)
            overlap_basis=mol_xc.intor("int1e_ovlp")
            mf=mol_xc.RHF().run(verbose=2,conv_tol=1e-8)
            true_HF_coeffmatrix= mf.mo_coeff[:, mf.mo_occ >=0.5] #Use all of them :)
            "re-calculate HF coefficients"
            overlaps=np.zeros(len(self.sample_points))
            for i in range(len(self.sample_points)):
                conv_sol=self.change_coefficients(self.HF_coefficients[i][:,:self.number_electronshalf],overlap_basis)
                determinant_matrix=self.getdeterminant_matrix(overlap_basis,true_HF_coeffmatrix,conv_sol)
                overlap=self.getoverlap(determinant_matrix)
                overlaps[i]=overlap
            overlap_to_HF[index]=np.dot(overlaps,self.eigvecs[index])
        return overlap_to_HF
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients):
        number_electronshalf=self.number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        Smat=np.zeros((len(self.HF_coefficients),len(self.HF_coefficients)))
        Tmat=np.zeros((len(self.HF_coefficients),len(self.HF_coefficients)))
        for i in range(len(self.HF_coefficients)):
            for j in range(i,len(self.HF_coefficients)):
                D,C_w,C_x=self.biorthogonalize(new_HF_coefficients[i],new_HF_coefficients[j],overlap_basis)
                D_tilde=D[np.abs(D)>1e-12]
                S_tilde=np.prod(D_tilde)**2 #Squared because this is RHF.
                S=np.prod(D)**2
                Smat[i,j]=Smat[j,i]=S
                if S<1e-10:
                    Smat[i,j]=Smat[j,i]=0
                energy=self.energy_func(C_w,C_x,energy_basis_1e,energy_basis_2e,D)*S_tilde
                #energy_naive=self.energy_func_naive(C_w,C_x,energy_basis_1e,energy_basis_2e,D)*S_tilde
                #print(energy-energy_naive)
                nuc_repulsion_energy=mol_xc.energy_nuc()*S
                Tmat[i,j]=Tmat[j,i]=energy+nuc_repulsion_energy
        return Smat,Tmat
    def energy_func_naive(self,C_w,C_x,onebodyp,twobodyp,D):
        twobody=ao2mo.get_mo_eri(twobodyp,(C_w,C_x,C_w,C_x),aosym="s1") #Works well and seems to me to be the correct one.
        onebody=contract("ki,lj,kl->ij",C_w,C_x,onebodyp)
        alpha=np.arange(3)
        beta=np.arange(3)
        onebodydivD=np.diag(onebody)/D
        onebody_alpha=np.sum(onebodydivD)
        onebody_beta=np.sum(onebodydivD)
        energy2=0
        #Equation 2.175 from Szabo, Ostlund
        for a in alpha:
            for b in alpha:
                e_contr=(twobody[a,a,b,b]-twobody[a,b,b,a])/(D[a]*D[b])
                energy2+=e_contr
        for a in alpha:
            for b in beta:
                e_contr=twobody[a,a,b,b]/(D[a]*D[b])
                energy2+=e_contr
        for a in beta:
            for b in alpha:
                e_contr=twobody[a,a,b,b]/(D[a]*D[b])
                energy2+=e_contr
        for a in beta:
            for b in beta:
                e_contr=(twobody[a,a,b,b]-twobody[a,b,b,a])/(D[a]*D[b])
                energy2+=e_contr
        energy2*=0.5
        return onebody_alpha+onebody_beta+energy2
    def energy_func(self,C_w,C_x,onebody,twobody,D):
        number_of_zeros=len(D[np.abs(D)<1e-12])*2
        zero_indices=np.where(np.abs(D)<1e-12)
        D_tilde=D[np.abs(D)>1e-12]
        S_tilde=np.prod(D_tilde)**2
        if number_of_zeros>2:
            return 0
        else:
            P_s=[]
            for i in range(self.number_electronshalf):
                P_s.append(contract("m,v->mv",C_w[:,i],C_x[:,i]))

            if number_of_zeros==2:
                H1=0
                i=zero_indices[0][0]
                J_contr=contract("sm,ms->",P_s[i],contract("tv,msvt->ms",P_s[i],twobody))
                #twobody=ao2mo.get_mo_eri(twobody,(C_w,C_x,C_w,C_x),aosym="s1")
                H2=J_contr#-K_contr
                #H2=S_tilde*(twobody[i,i,i,i])
            elif number_of_zeros==0:
                P=P_s.copy()
                for i in range(self.number_electronshalf):
                    P[i]=P[i]/D[i]
                W=np.sum(P,axis=0)
                H2=0.5*contract("sm,ms->",W,4*contract("tv,msvt->ms",W,twobody)-2*contract("tv,mtvs->ms",W,twobody))
                #K_contr=2*contract("sm,ms->",W,)
                #H2=0.5*(J_contr-K_contr)
                H1=2*contract("mv,mv->",W,onebody)
            return H2+H1
class eigvecsolver_RHF_singlesdoubles(eigvecsolver_RHF):
    def data_creator(self):
        neh=self.number_electronshalf
        num_bas=self.num_bas
        states=self.state_creator()
        try:
            self.nonzeros
        except:
            self.nonzeros=None
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
        neh=self.number_electronshalf
        groundstring="1"*neh+"0"*(self.n_unocc) #Ground state Slater determinant
        alpha_singleexcitations=[]
        for i in range(neh):
            for j in range(neh,self.num_bas):
                newstring=groundstring[:i]+"0"+groundstring[i+1:j]+"1"+groundstring[j+1:] #Take the ith 1 and move it to position j
                alpha_singleexcitations.append(newstring)
        if self.doubles:
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
        if self.doubles:
            doubles_alpha=[[alpha,groundstring] for alpha in alpha_doubleexcitations]
            doubles_beta=[[groundstring,alpha] for alpha in alpha_doubleexcitations]
            doubles_alphabeta=[[alpha,beta] for alpha in alpha_singleexcitations for beta in alpha_singleexcitations]
            allstates=GS+singles_alpha+singles_beta+doubles_alpha+doubles_beta+doubles_alphabeta
        else:
            allstates=GS+singles_alpha+singles_beta
        return allstates
    def index_creator(self):
        all_indices=[]
        for state in self.states:
            alphas_occ=[i for i in range(len(state[0])) if int(state[0][i])==1]
            betas_occ=[i for i in range(len(state[1])) if int(state[1][i])==1]
            all_indices.append([alphas_occ,betas_occ])
        return all_indices
    def calculate_energies(self,xc_array,doubles=True):
        self.doubles=doubles
        self.data_creator()
        """Calculates the molecule's energy"""
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        self.all_new_HF_coefficients=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.sample_points)):
                new_HF_coefficients.append(self.change_coefficients(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))) #Actually take the WHOLE thing.
                #new_HF_coefficients.append(self.change_coefficients(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf])
            self.all_new_HF_coefficients.append(new_HF_coefficients)
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            try:
                eigval,eigvec=generalized_eigenvector(T,S)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval

            eigval_array.append(eigvec)
        return energy_array,eigval_array
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients,threshold=1e-12):
        number_electronshalf=self.number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        Smat=np.zeros((len(self.HF_coefficients),len(self.HF_coefficients)))
        Tmat=np.zeros((len(self.HF_coefficients),len(self.HF_coefficients)))
        n_HF_coef=len(new_HF_coefficients)
        num_states=len(self.states)
        number_matrix_elements=n_HF_coef*num_states
        Smat=np.zeros((number_matrix_elements,number_matrix_elements))
        Tmat=np.zeros((number_matrix_elements,number_matrix_elements))
        for c1 in range(number_matrix_elements): #For each matrix element
            print("%d/%d"%(c1,number_matrix_elements))
            i=c1//num_states # The number of the HF determinant
            j=c1%num_states #Number of the excitation
            C_w_a=new_HF_coefficients[i][:,self.indices[j][0]]
            C_w_b=new_HF_coefficients[i][:,self.indices[j][1]]
            for c2 in range(c1,number_matrix_elements):
                k=c2//num_states
                l=c2%num_states
                C_x_a=new_HF_coefficients[k]
                C_x_a=C_x_a[:,self.indices[l][0]]
                C_x_b=new_HF_coefficients[k][:,self.indices[l][1]]

                D_a,C_w_a,C_x_a=self.biorthogonalize(C_w_a,C_x_a,overlap_basis)
                D_b,C_w_b,C_x_b=self.biorthogonalize(C_w_b,C_x_b,overlap_basis)
                D_tilde_a=D_a[np.abs(D_a)>threshold]
                D_tilde_b=D_b[np.abs(D_b)>threshold]
                S_tilde=np.prod(D_tilde_a)*np.prod(D_tilde_b) #Squared because this is RHF.
                S=S_tilde*(len(D_tilde_b)==len(D_b))*(len(D_tilde_a)==len(D_a))
                Smat[c1,c2]=Smat[c2,c1]=S
                energy=self.energy_func(C_w_a,C_w_b,C_x_a,C_x_b,energy_basis_1e,energy_basis_2e,D_a,D_b,c1,c2)*S_tilde
                nuc_repulsion_energy=mol_xc.energy_nuc()*S
                Tmat[c1,c2]=Tmat[c2,c1]=energy+nuc_repulsion_energy
        print(np.max(np.abs(Tmat)))
        return Smat,Tmat
    def energy_func_naive(self,C_w_a,C_w_b,C_x_a,C_x_b,onebodyp,twobodyp,D_a,D_b,c1,c2):
        number_of_zeros_alpha=len(D_a[np.abs(D_a)<1e-12])
        number_of_zeros_beta=len(D_a[np.abs(D_b)<1e-12])
        number_of_zeros=number_of_zeros_alpha+number_of_zeros_beta
        twobody_aaaa=ao2mo.get_mo_eri(twobodyp,(C_w_a,C_x_a,C_w_a,C_x_a),aosym="s1")
        twobody_bbbb=ao2mo.get_mo_eri(twobodyp,(C_w_b,C_x_b,C_w_b,C_x_b),aosym="s1")
        twobody_aabb=ao2mo.get_mo_eri(twobodyp,(C_w_a,C_x_a,C_w_b,C_x_b),aosym="s1")
        twobody_bbaa=ao2mo.get_mo_eri(twobodyp,(C_w_b,C_x_b,C_w_a,C_x_a),aosym="s1")
        onebody_alpha=contract("ki,lj,kl->ij",C_w_a,C_x_a,onebodyp)
        onebody_beta=contract("ki,lj,kl->ij",C_w_b,C_x_b,onebodyp)
        alpha=np.arange(self.number_electronshalf)
        beta=np.arange(self.number_electronshalf)

        #Equation 2.175 from Szabo, Ostlund

        if number_of_zeros>2:
            return 0
        zero_indices_alpha=np.where(np.abs(D_a)<1e-12)
        zero_indices_beta=np.where(np.abs(D_b)<1e-12)
        H1=0; H2=0
        if number_of_zeros==2:
            if number_of_zeros_alpha==2:
                i=zero_indices_alpha[0][0]
                j=zero_indices_alpha[0][1]
                H2=twobody_aaaa[i,i,j,j]-twobody_aaaa[i,j,j,i]
            elif number_of_zeros_beta==2:
                i=zero_indices_beta[0][0]
                j=zero_indices_beta[0][1]
                H2=twobody_bbbb[i,i,j,j]-twobody_bbbb[i,j,j,i]
            else:
                i=zero_indices_alpha[0][0]
                j=zero_indices_beta[0][0]
                H2=twobody_aabb[i,i,j,j]

        elif number_of_zeros_alpha==1:
            i=zero_indices_alpha[0][0]
            H1=onebody_alpha[i,i]
            for j in alpha:
                if i==j:
                    continue
                H2+=(twobody_aaaa[i,i,j,j]-twobody_aaaa[i,j,j,i])/D_a[j]
            for j in beta:
                H2+=twobody_aabb[i,i,j,j]/D_b[j]
        elif number_of_zeros_beta==1:

            i=zero_indices_beta[0][0]
            H1=onebody_beta[i,i]
            for j in beta:
                if i==j:
                    continue
                H2+=(twobody_bbbb[i,i,j,j]-twobody_bbbb[i,j,j,i])/D_b[j]
            for j in alpha:
                H2+=twobody_bbaa[i,i,j,j]/D_a[j]

        elif number_of_zeros==0:
            onebody_alphav=np.sum(np.diag(onebody_alpha)/D_a)
            onebody_betav=np.sum(np.diag(onebody_beta)/D_b)
            energy2=0
            for a in alpha:
                for b in alpha:
                    e_contr=(twobody_aaaa[a,a,b,b]-twobody_aaaa[a,b,b,a])#/(D_a[a]*D_a[b])
                    energy2+=e_contr
            for a in alpha:
                for b in beta:
                    e_contr=twobody_aabb[a,a,b,b]#/(D_a[a]*D_b[b])
                    energy2+=e_contr
            for a in beta:
                for b in alpha:
                    e_contr=twobody_bbaa[a,a,b,b]#/(D_b[a]*D_a[b])
                    energy2+=e_contr
            for a in beta:
                for b in beta:
                    e_contr=(twobody_bbbb[a,a,b,b]-twobody_bbbb[a,b,b,a])#/(D_b[a]*D_b[b])
                    energy2+=e_contr
            energy2*=0.5
            H1=onebody_alphav+onebody_betav
            H2=energy2
        return H2+H1
    def energy_func(self,C_w_a,C_w_b,C_x_a,C_x_b,onebody,twobody,D_a,D_b,c1=0,c2=0,threshold=1e-12):
        number_of_zeros_alpha=len(D_a[np.abs(D_a)<threshold])
        number_of_zeros_beta=len(D_a[np.abs(D_b)<threshold])
        number_of_zeros=number_of_zeros_alpha+number_of_zeros_beta
        if number_of_zeros>2:
            return 0
        zero_indices_alpha=np.where(np.abs(D_a)<threshold)
        zero_indices_beta=np.where(np.abs(D_b)<threshold)
        P_s_a=[]
        for i in range(self.number_electronshalf):
            P_s_a.append(contract("m,v->mv",C_w_a[:,i],C_x_a[:,i]))
        P_s_b=[]
        for i in range(self.number_electronshalf):
            P_s_b.append(contract("m,v->mv",C_w_b[:,i],C_x_b[:,i]))
        P_a=P_s_a.copy()
        for i in range(self.number_electronshalf):
            if np.abs(D_a[i])>1e-12:
                P_a[i]=P_a[i]/D_a[i]
            else:
                P_a[i]=0*P_a[i] # No contribution to W as we can ignore the summation over i
        W_a=np.sum(P_a,axis=0)
        P_b=P_s_b.copy()
        for i in range(self.number_electronshalf):
            if np.abs(D_b[i])>1e-12:
                P_b[i]=P_b[i]/D_b[i]
            else:
                P_b[i]=0*P_b[i] # No contribution to W as we can ignore the summation over i
        W_b=np.sum(P_b,axis=0)
        H1=H2=0
        if number_of_zeros==2:
            if number_of_zeros_alpha==2:
                i=zero_indices_alpha[0][0]
                j=zero_indices_alpha[0][1]
                H2=contract("sm,ms->",P_s_a[i],contract("tv,msvt->ms",P_s_a[j],twobody)-contract("tv,mtvs->ms",P_s_a[j],twobody))
            elif number_of_zeros_beta==2:
                i=zero_indices_beta[0][0]
                j=zero_indices_beta[0][1]
                H2=contract("sm,ms->",P_s_b[i],contract("tv,msvt->ms",P_s_b[j],twobody)-contract("tv,mtvs->ms",P_s_b[j],twobody))
            else:
                i=zero_indices_alpha[0][0]
                j=zero_indices_beta[0][0]
                J_contr=contract("sm,ms->",P_s_a[i],contract("tv,msvt->ms",P_s_b[j],twobody))
                H2=J_contr
        elif number_of_zeros_alpha==1:
            i=zero_indices_alpha[0][0]
            H1=contract("mv,mv->",P_s_a[i],onebody)
            J_contr=contract("sm,ms->",P_s_a[i],contract("tv,msvt->ms",W_a+W_b,twobody)-contract("tv,mtvs->ms",W_a,twobody))#+contract("sm,ms->",P_s_a[i],contract("tv,msvt->ms",W_b,twobody))
            H2=J_contr
        elif number_of_zeros_beta==1:
            i=zero_indices_beta[0][0]
            H1=contract("mv,mv->",P_s_b[i],onebody)
            J_contr=contract("sm,ms->",P_s_b[i],contract("tv,msvt->ms",W_a+W_b,twobody)-contract("tv,mtvs->ms",W_b,twobody))#+contract("sm,ms->",P_s_b[i],contract("tv,msvt->ms",W_b,twobody))
            H2=J_contr
        elif number_of_zeros==0:
            W=W_a+W_b
            J_a_a=contract("sm,ms->",W,contract("tv,msvt->ms",W,twobody))
            J_contr=J_a_a
            K_contr=contract("sm,ms->",W_a,contract("tv,mtvs->ms",W_a,twobody))+contract("sm,ms->",W_b,contract("tv,mtvs->ms",W_b,twobody))
            H2=0.5*(J_contr-K_contr)
            H1=contract("mv,mv->",W,onebody)
        return H2+H1
class eigvecsolver_RHF_singles(eigvecsolver_RHF_singlesdoubles):
    def data_creator(self):
        neh=self.number_electronshalf
        num_bas=self.num_bas
        states=self.state_creator()
        try:
            self.nonzeros
        except:
            self.nonzeros=None
        if self.nonzeros is not None:
            self.states=[states[i] for i in self.nonzeros]
        else:
            self.states=states
        indices=np.array(self.index_creator())
        self.num_states=len(self.states)
        self.indices=indices
    def state_creator(self):
        neh=self.number_electronshalf
        groundstring="1"*neh+"0"*(self.n_unocc) #Ground state Slater determinant, the only one which is just ONE determinant...
        alpha_singleexcitations=[]
        for i in range(neh):
            for j in range(neh,self.num_bas):
                newstring=groundstring[:i]+"0"+groundstring[i+1:j]+"1"+groundstring[j+1:] #Take the ith 1 and move it to position j
                alpha_singleexcitations.append(newstring)
        GS=[[groundstring,groundstring]]
        singles_alpha=[[alpha,groundstring] for alpha in alpha_singleexcitations] #All single excitations within alpha
        singles_beta=[[groundstring,alpha] for alpha in alpha_singleexcitations]
        singles_spinstate=[]
        for i in range(len(singles_alpha)):
            singles_spinstate.append([singles_alpha[i],singles_beta[i]]) #Might need to keep track of the sign? Well...
        allstates=[GS]+singles_spinstate
        return allstates
    def index_creator(self):
        all_indices=[]
        for state in self.states: #For each CAS state
            occ_state=[]
            for staterino in range(len(state)): #For each Slater determinant
                alphas_occ=[i for i in range(len(state[staterino][0])) if int(state[staterino][0][i])==1]
                betas_occ=[i for i in range(len(state[staterino][1])) if int(state[staterino][1][i])==1]
                occ_state.append([alphas_occ,betas_occ])
            all_indices.append(occ_state)
        return all_indices
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients,threshold=1e-12):
        number_electronshalf=self.number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        n_HF_coef=len(new_HF_coefficients)
        num_states=len(self.states)
        number_matrix_elements=n_HF_coef*num_states
        Smat=np.zeros((number_matrix_elements,number_matrix_elements))
        Tmat=np.zeros((number_matrix_elements,number_matrix_elements))
        for c1 in range(number_matrix_elements): #For each matrix element
            print("%d/%d"%(c1,number_matrix_elements))
            i=c1//num_states # Number of the reference state
            j=c1%num_states #Number of the excitation
            for c2 in range(c1,number_matrix_elements):
                k=c2//num_states
                l=c2%num_states
                for casLeft in range(len(self.states[j])):
                    for casRight in range(len(self.states[l])):
                        C_w_a=new_HF_coefficients[i][:,self.indices[j][casLeft][0]]
                        C_w_b=new_HF_coefficients[i][:,self.indices[j][casLeft][1]]
                        C_x_a=new_HF_coefficients[k][:,self.indices[l][casRight][0]]
                        C_x_b=new_HF_coefficients[k][:,self.indices[l][casRight][1]]
                        D_a,C_w_a,C_x_a=self.biorthogonalize(C_w_a,C_x_a,overlap_basis)
                        D_b,C_w_b,C_x_b=self.biorthogonalize(C_w_b,C_x_b,overlap_basis)

                        D_tilde_a=D_a[np.abs(D_a)>threshold]
                        D_tilde_b=D_b[np.abs(D_b)>threshold]
                        S_tilde=np.prod(D_tilde_a)*np.prod(D_tilde_b)
                        S=S_tilde*(len(D_tilde_b)==len(D_b))*(len(D_tilde_a)==len(D_a))

                        Smat[c1,c2]+=S
                        if(c1 != c2):
                            Smat[c2,c1]+=S
                        energy=self.energy_func(C_w_a,C_w_b,C_x_a,C_x_b,energy_basis_1e,energy_basis_2e,D_a,D_b,c1,c2)*S_tilde
                        nuc_repulsion_energy=mol_xc.energy_nuc()*S
                        Tmat[c1,c2]+=energy+nuc_repulsion_energy
                        if(c1 != c2):
                            Tmat[c2,c1]+=energy+nuc_repulsion_energy
                reducement=np.sqrt(len(self.states[j])*len(self.states[l]))
                Tmat[c1,c2]/=reducement
                if(c1 != c2):
                    Tmat[c2,c1]/=reducement
                Smat[c1,c2]/=reducement
                if(c1 != c2):
                    Smat[c2,c1]/=reducement
        print(np.max(np.abs(Tmat)))
        return Smat,Tmat
class eigensolver_RHF_knowncoefficients(eigvecsolver_RHF):
    def __init__(self,sample_coefficients,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='Bohr',charge=0,symmetry=False):
        """Initiate the solver.
        It Creates the HF coefficient matrices for the sample points for a given molecule.

        Input:
        sample_points (array) - the points at which to evaluate the WF
        basis_type - atomic basis
        molecule - A function f(x), returning the molecule as function of x
        """
        self.molecule=molecule
        self.basis_type=basis_type
        self.spin=spin
        self.unit=unit
        self.charge=charge
        self.symmetry=symmetry
        self.HF_coefficients=sample_coefficients #An interesting observation here is that the basis does not matter
        self.build_molecule(1) #Initiate number of electrons, that's literally the only purpose here.
    def calculate_energies(self,xc_array,adapt_geometry=False):
        """Calculates the molecule's energy"""
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            if adapt_geometry:
                for i in range(len(self.HF_coefficients)):
                    new_HF_coefficients.append(self.change_coefficients(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf])
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            try:
                eigval,eigvec=generalized_eigenvector(T,S)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval
            eigval_array.append(eigval)
        return energy_array,eigval_array
class eigvecsolver_RHF_coupling(eigvecsolver_RHF):
    def __init__(self,sample_lambdas,sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False):
        self.sample_positions=sample_points
        self.HF_coefficients=[] #The Hartree Fock coefficients solved at the sample points
        self.molecule=molecule
        self.basis_type=basis_type
        self.spin=spin
        self.unit=unit
        self.charge=charge
        self.symmetry=symmetry
        self.sample_points=sample_lambdas
    def solve_HF(self,sample_point):
        """Solve equations for different RHF's"""
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(sample_point)
            mf = scf.RHF(mol)
            eri=mol.intor('int2e',aosym="s1")*x
            mf._eri = ao2mo.restore(1,eri,mol.nao_nr())
            mol.incore_anyway=True
            mf.kernel()
            expansion_coefficients_mol= mf.mo_coeff[:, mf.mo_occ > 0.]
            HF_coefficients.append(expansion_coefficients_mol)
        self.HF_coefficients=HF_coefficients
    def calculate_energies(self,xc_array):
            """Calculates the molecule's energy"""
            energy_array=np.zeros(len(xc_array))
            eigval_array=[]
            for index,xc in enumerate(xc_array):
                self.solve_HF(xc) #Update HF coefficients
                mol_xc=self.build_molecule(xc)
                S,T=self.calculate_ST_matrices(mol_xc,self.HF_coefficients)
                try:
                    eigval,eigvec=generalized_eigenvector(T,S,threshold=1e-12)
                except:
                    eigval=float('NaN')
                    eigvec=float('NaN')
                energy_array[index]=eigval
                eigval_array.append(eigval)
            return energy_array,eigval_array
