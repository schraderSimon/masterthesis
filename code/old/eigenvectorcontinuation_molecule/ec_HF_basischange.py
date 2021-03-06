import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import numba as nb
from pyscf import gto, scf, fci,cc,ao2mo, mp, mcscf
import pyscf
import sys
import itertools
import scipy
from eigenvectorcontinuation import *
np.set_printoptions(linewidth=200,precision=2,suppress=True)
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)
class eigvecsolver_RHF():
    def __init__(self,sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False):
        """Initiate the solver.
        It Creates the HF coefficient matrices for the sample points for a given molecule.

        Input:
        sample_points (array) - the points at which to evaluate the WF
        basis_type - atomic basis
        molecule - A function f(x), returning the molecule as function of x
        """
        self.HF_coefficients=[] #The Hartree Fock coefficients solved at the sample points
        self.molecule=molecule
        self.basis_type=basis_type
        self.spin=spin
        self.unit=unit
        self.symmetry=symmetry
        self.charge=charge
        self.sample_points=sample_points
        self.solve_HF()
    def solve_HF(self):
        """
        Create coefficient matrices for each sample point
        """
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(x)
            mf=mol.RHF().run(verbose=2,conv_tol=1e-8)
            expansion_coefficients_mol= mf.mo_coeff[:, mf.mo_occ >=-1] #Use all
            HF_coefficients.append(expansion_coefficients_mol)
        self.HF_coefficients=HF_coefficients
        self.nMO_h=HF_coefficients[0].shape[0] #Number of molecular orbitals divided by two (for each spin)
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
        eigval_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.sample_points)):
                new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf])
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            try:
                eigval,eigvec=generalized_eigenvector(T,S)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval
            eigval_array.append(eigval)
        return energy_array,eigval_array
    def getdeterminant_matrix(self,AO_overlap,HF_coefficients_left,HF_coefficients_right):
        determinant_matrix=np.einsum("ab,ai,bj->ij",AO_overlap,HF_coefficients_left,HF_coefficients_right)
        return determinant_matrix
    def getoverlap(self,determinant_matrix):
        overlap=np.linalg.det(determinant_matrix)**2
        return overlap
    def onebody_energy(self,energy_basis_1e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
        Hamiltonian_SLbasis=np.einsum("ki,lj,kl->ij",HF_coefficients_left,HF_coefficients_right,energy_basis_1e)
        energy_1e=0
        for k in range(self.number_electronshalf):
            determinant_matrix_energy=determinant_matrix.copy() #Re-initiate Energy matrix
            for l in range(self.number_electronshalf):
                    determinant_matrix_energy[l,k]=Hamiltonian_SLbasis[l,k]
            energy_contribution=np.linalg.det(determinant_matrix_energy)*np.linalg.det(determinant_matrix)
            energy_1e+=energy_contribution
        energy_1e*=2 #Beta spin part
        return energy_1e

    def twobody_energy(self,energy_basis_2e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
        MO_eri=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left,HF_coefficients_right,HF_coefficients_left,HF_coefficients_right))
        energy_2e=0
        n=int(self.number_electronshalf*2)
        nh=self.number_electronshalf
        G1s,G2s=get_antisymm_element_separated_RHF(MO_eri,int(n))
        M1s,M2s=second_order_adj_matrix_blockdiag_separated_RHF(determinant_matrix)
        energy_2e=2*np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))
        return energy_2e
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients):
        number_electronshalf=self.number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        S=np.zeros((len(self.HF_coefficients),len(self.HF_coefficients)))
        T=np.zeros((len(self.HF_coefficients),len(self.HF_coefficients)))
        for i in range(len(self.HF_coefficients)):
            for j in range(i,len(self.HF_coefficients)):
                determinant_matrix=self.getdeterminant_matrix(overlap_basis,new_HF_coefficients[i],new_HF_coefficients[j])
                overlap=self.getoverlap(determinant_matrix)
                S[i,j]=S[j,i]=overlap
                nuc_repulsion_energy=mol_xc.energy_nuc()*overlap
                energy_1e=self.onebody_energy(energy_basis_1e,new_HF_coefficients[i],new_HF_coefficients[j],determinant_matrix)
                energy_2e=self.twobody_energy(energy_basis_2e,new_HF_coefficients[i],new_HF_coefficients[j],determinant_matrix)
                energy_total=energy_2e+energy_1e+nuc_repulsion_energy
                T[i,j]=energy_total
                T[j,i]=energy_total
        return S,T
    def basischange(self,C_old,overlap_AOs_newnew):
        S_eig,S_U=np.linalg.eigh(overlap_AOs_newnew)
        S_poweronehalf=S_U@np.diag(S_eig**0.5)@S_U.T
        S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
        C_newbasis=S_poweronehalf@C_old #Basis change
        q,r=np.linalg.qr(C_newbasis) #orthonormalise
        return S_powerminusonehalf@q #change back

    def basischange_alt(self,C_old,overlap_AOs_newnew):
        S=self.getdeterminant_matrix(overlap_AOs_newnew,C_old,C_old)
        S_eig,S_U=np.linalg.eigh(S)
        S_poweronehalf=S_U@np.diag(S_eig**0.5)@S_U.T
        S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
        return S_powerminusonehalf@C_old
class eigvecsolver_RHF_singles(eigvecsolver_RHF):

    def create_singles(self,expansion_coefficients):
        """
        Here, I create all possible singles within the basis set, and then return the basis set.
        To get the correct, spin-adapted singles, the following things need to be done:
        I need to couple the spacial singles with the spacial ground state...
        ...and then couple the spacial ground state with a single, such that the correct CAS is created.
        """
        basisset_size=len(expansion_coefficients[0][:,0])
        n_occ=self.number_electronshalf
        n_unocc=basisset_size-n_occ
        permutations=[]
        permutations.append(np.array([0,0])) #No-swap-operator
        #1. Create all possible permutations
        for i in range(n_occ):
            for j in range(n_occ,basisset_size):
                permutations.append(np.array([i,j])) #This means: i out, j in!
        #Check: Correct.
        return permutations
    def calculate_energies(self,xc_array):
        """Calculates the molecule's energy"""
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.sample_points)):
                new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:]) #Actually take the WHOLE thing.
                #new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf])
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            try:
                eigval,eigvec=generalized_eigenvector(T,S)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval
            eigval_array.append(eigval)
        return energy_array,eigval_array
    def getoverlap(self,determinant_matrix):
        overlap=np.linalg.det(determinant_matrix[0])*np.linalg.det(determinant_matrix[1]) #alpha part times beta part
        return overlap
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients):
        number_electronshalf=self.number_electronshalf
        neh=number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        permutations=self.create_singles(new_HF_coefficients)
        number_matrix_elements=len(self.HF_coefficients)*(len(permutations))
        S=np.zeros((number_matrix_elements,number_matrix_elements))
        T=np.zeros((number_matrix_elements,number_matrix_elements))
        e1_slbasis=[]
        e2_slbasis=[]

        """Find the 1 and 2 electron integrals in the extended basis"""
        for i in range(len(self.HF_coefficients)):
            temp_1=[]
            temp_2=[]
            for k in range(len(self.HF_coefficients)):
                temp_1.append(np.einsum("ki,lj,kl->ij",new_HF_coefficients[i],new_HF_coefficients[k],energy_basis_1e))
                basis=ao2mo.get_mo_eri(energy_basis_2e,(new_HF_coefficients[i],new_HF_coefficients[k],new_HF_coefficients[i],new_HF_coefficients[k]),aosym="s1")
                temp_2.append(basis)
            e1_slbasis.append(temp_1)
            e2_slbasis.append(temp_2)
        len_permutations=len(permutations)
        for c1 in range(number_matrix_elements):
            i=c1//len_permutations
            j=c1%len_permutations
            for c2 in range(c1,number_matrix_elements):
                determinant_matrix_groundstate_alpha=self.getdeterminant_matrix(overlap_basis,new_HF_coefficients[i][:,:neh],new_HF_coefficients[k][:,:neh]) #The unmodified part
                k=c2//len_permutations
                l=c2%len_permutations
                energy_1e=0
                energy_2e=0
                if l==0 and j==0: #Both are non-excited states
                    determinant=self.getdeterminant_matrix(overlap_basis,new_HF_coefficients[i][:,:neh],new_HF_coefficients[k][:,:neh])
                    overlap=self.getoverlap([determinant,determinant])
                    [pl,pr],[nana,nene]=self.getpermutations(permutations,(0,j),(0,l),(0,0),(0,0)) #No permutations are performed, this is to work with the functions
                    energy_1e=self.onebody_energy([determinant,determinant],e1_slbasis[i][k],pl,pr)
                    energy_2e=self.twobody_energy([determinant,determinant],e2_slbasis[i][k],pl,pr)

                    S[c1,c2]=S[c2,c1]=overlap
                    #Calculate energy of CAS.
                    nuc_repulsion_energy=mol_xc.energy_nuc()*overlap
                    energy_total=energy_2e+energy_1e+nuc_repulsion_energy
                    T[c1,c2]=T[c2,c1]=energy_total
                    continue
                GS_Left=new_HF_coefficients[i][:,:neh]
                GS_Right=new_HF_coefficients[k][:,:neh]
                ES_Left=swap_cols(new_HF_coefficients[i],permutations[j][0],permutations[j][1])[:,:neh]
                ES_Right=swap_cols(new_HF_coefficients[k],permutations[l][0],permutations[l][1])[:,:neh]
                determinant_matrix_GS_GS=self.getdeterminant_matrix(overlap_basis,GS_Left,GS_Right) #The unmodified part
                determinant_matrix_EX_EX=self.getdeterminant_matrix(overlap_basis,ES_Left,ES_Right)
                determinant_matrix_GS_EX=self.getdeterminant_matrix(overlap_basis,GS_Left,ES_Right) #The unmodified part
                determinant_matrix_EX_GS=self.getdeterminant_matrix(overlap_basis,ES_Left,GS_Right)
                if j==0: #Left state is ground state
                    multiplier=1/np.sqrt(2)
                    determinant_matrix_1=[determinant_matrix_GS_GS,determinant_matrix_GS_EX]
                    determinant_matrix_2=[determinant_matrix_GS_EX,determinant_matrix_GS_GS]
                    permutations_loc=self.getpermutations(permutations,(0,j),(0,l),(j,0),(l,0))
                elif l==0: #Right state is ground state
                    multiplier=1/np.sqrt(2)
                    determinant_matrix_1=[determinant_matrix_GS_GS,determinant_matrix_EX_GS]
                    determinant_matrix_2=[determinant_matrix_EX_GS,determinant_matrix_GS_GS]
                    permutations_loc=self.getpermutations(permutations,(0,j),(0,l),(j,0),(l,0))
                else:
                    multiplier=1
                    determinant_matrix_1=[determinant_matrix_GS_GS,determinant_matrix_EX_EX]
                    determinant_matrix_2=[determinant_matrix_GS_EX,determinant_matrix_EX_GS]
                    permutations_loc=self.getpermutations(permutations,(0,j),(0,l),(0,j),(l,0))

                determinant_matrix=[determinant_matrix_1,determinant_matrix_2]
                overlap=(self.getoverlap(determinant_matrix_1)+self.getoverlap(determinant_matrix_2))*multiplier
                S[c2,c1]=S[c1,c2]=overlap
                nuc_repulsion_energy=mol_xc.energy_nuc()*overlap/multiplier

                for ind,(pl, pr) in enumerate(permutations_loc):
                    energy_1e+=self.onebody_energy(determinant_matrix[ind],e1_slbasis[i][k],pl,pr)
                    energy_2e+=self.twobody_energy(determinant_matrix[ind],e2_slbasis[i][k],pl,pr)
                energy_total=(energy_2e+energy_1e+nuc_repulsion_energy)*multiplier
                T[c1,c2]=T[c2,c1]=energy_total#=T[k*(len(permutations))+l,c1]=energy_total
        print("Done one")
        return S,T
    def getpermutations(self,permutations,l1,r1,l2,r2):
        permutations_left_1=[permutations[l1[0]],permutations[l1[1]]]
        permutations_right_1=[permutations[r1[0]],permutations[r1[1]]]
        permutations_left_2=[permutations[l2[0]],permutations[l2[1]]]
        permutations_right_2=[permutations[r2[0]],permutations[r2[1]]]
        return [[permutations_left_1,permutations_right_1],[permutations_left_2,permutations_right_2]]
    def onebody_energy_alt(self,determinant_matrix,SLbasis,permutations_left,permutations_right):
        """Unlike the previous method, the left and right coefficients are now the WHOLE set (occ+unocc). The SLbasis is also the whole (for the corresponding system). The permutations contain all information
        about the construction of the coefficients.
        """

        neh=self.number_electronshalf
        electrons_basis=np.arange(neh)
        electrons_alpha_left=np.where(electrons_basis==permutations_left[0][0],permutations_left[0][1],electrons_basis)
        electrons_beta_left=np.where(electrons_basis==permutations_left[1][0],permutations_left[1][1],electrons_basis)
        electrons_alpha_right=np.where(electrons_basis==permutations_right[0][0],permutations_right[0][1],electrons_basis)
        electrons_beta_right=np.where(electrons_basis==permutations_right[1][0],permutations_right[1][1],electrons_basis)
        Hamiltonian_SLbasis_alpha=SLbasis[np.ix_(electrons_alpha_left,electrons_alpha_right)]
        Hamiltonian_SLbasis_beta=SLbasis[np.ix_(electrons_beta_left,electrons_beta_right)]
        energy_1e=0
        determinant_matrix_alpha=determinant_matrix[0]
        determinant_matrix_beta=determinant_matrix[1]
        for k in range(neh):
            determinant_matrix_energy_alpha=determinant_matrix_alpha.copy() #Re-initiate Energy matrix
            for l in range(neh):
                    determinant_matrix_energy_alpha[l,k]=Hamiltonian_SLbasis_alpha[l,k]
            energy_contribution=np.linalg.det(determinant_matrix_energy_alpha)*np.linalg.det(determinant_matrix_beta)
            energy_1e+=energy_contribution
        for k in range(neh):
            determinant_matrix_energy_beta=determinant_matrix_beta.copy() #Re-initiate Energy matrix
            for l in range(neh):
                    determinant_matrix_energy_beta[l,k]=Hamiltonian_SLbasis_beta[l,k]
            energy_contribution=np.linalg.det(determinant_matrix_energy_beta)*np.linalg.det(determinant_matrix_alpha)
            energy_1e+=energy_contribution
        return energy_1e
    def onebody_energy(self,determinant_matrix,SLbasis,permutations_left,permutations_right):
        """Unlike the previous method, the left and right coefficients are now the WHOLE set (occ+unocc). The SLbasis is also the whole (for the corresponding system). The permutations contain all information
        about the construction of the coefficients.
        """
        threshold=1e-5
        Linv_a,da,Rinv_a=LDU_decomp(determinant_matrix[0])#,threshold=threshold) #Alpha LdR decomposition
        Linv_b,db,Rinv_b=LDU_decomp(determinant_matrix[1])#,threshold=threshold) #Beta LdR decomposition
        neh=self.number_electronshalf
        num_singularities_left=len(da[np.abs(da)<threshold])
        num_singularities_right=len(db[np.abs(db)<threshold])
        num_singularities=num_singularities_left+num_singularities_right
        if num_singularities>=2:
            return 0
        La=Linv_a.T
        Lb=Linv_b.T
        Ra=Rinv_a.T
        Rb=Rinv_b.T
        electrons_basis=np.arange(neh)
        electrons_alpha_left=np.where(electrons_basis==permutations_left[0][0],permutations_left[0][1],electrons_basis)
        electrons_beta_left=np.where(electrons_basis==permutations_left[1][0],permutations_left[1][1],electrons_basis)
        electrons_alpha_right=np.where(electrons_basis==permutations_right[0][0],permutations_right[0][1],electrons_basis)
        electrons_beta_right=np.where(electrons_basis==permutations_right[1][0],permutations_right[1][1],electrons_basis)
        Hamiltonian_SLbasis_alpha=SLbasis[np.ix_(electrons_alpha_left,electrons_alpha_right)]
        Hamiltonian_SLbasis_beta=SLbasis[np.ix_(electrons_beta_left,electrons_beta_right)]
        energy_1e=0.0
        if num_singularities==1:
            if num_singularities_left==1:
                val=np.prod(da[np.abs(da)>threshold])*np.prod(db[np.abs(db)>threshold])
                adj_alpha=np.outer(Ra[:,neh-1],La[neh-1,:])*val
                energy_1e=np.trace(Hamiltonian_SLbasis_alpha@adj_alpha)
                assert np.abs(energy_1e-self.onebody_energy_alt(determinant_matrix,SLbasis,permutations_left,permutations_right))<1e-10,"left"
            elif num_singularities_right==1:
                val=np.prod(da[np.abs(da)>threshold])*np.prod(db[np.abs(db)>threshold])
                adj_beta=np.outer(Rb[:,-1],Lb[-1,:])*val
                energy_1e=np.trace(Hamiltonian_SLbasis_beta@adj_beta)
                assert np.abs(energy_1e-self.onebody_energy_alt(determinant_matrix,SLbasis,permutations_left,permutations_right))<1e-10,"right"
        else:
            adj_alpha=first_order_adj_matrix_LdR(La,da,Ra)
            adj_beta=first_order_adj_matrix_LdR(Lb,db,Rb)
            energy_1e_1=np.trace(Hamiltonian_SLbasis_alpha@adj_alpha)+np.trace(Hamiltonian_SLbasis_beta@adj_beta)
            full=scipy.linalg.block_diag(adj_alpha,adj_beta)
            energy_1e_2=np.trace(full@scipy.linalg.block_diag(Hamiltonian_SLbasis_alpha,Hamiltonian_SLbasis_beta))
            true_e=self.onebody_energy_alt(determinant_matrix,SLbasis,permutations_left,permutations_right)
            try:
                assert np.abs(energy_1e_1-true_e)<1,"inbetween %f"%(energy_1e_1-true_e)
            except:
                assert(np.all(np.abs(Linv_a@np.diag(da)@Rinv_a-determinant_matrix[0])<1e-10)),"eeype"
                assert(np.all(np.abs(Linv_b@np.diag(db)@Rinv_b-determinant_matrix[1])<1e-10)),"eeype"
                determinant_matrix_alpha=determinant_matrix[0]
                determinant_matrix_beta=determinant_matrix[1]
                M=np.zeros(determinant_matrix[0].shape)
                for k in range(neh):
                    determinant_matrix_energy_alpha=determinant_matrix_alpha.copy() #Re-initiate Energy matrix
                    for l in range(neh):
                            determinant_matrix_energy_alpha[l,k]=Hamiltonian_SLbasis_alpha[l,k]
                    print(np.linalg.det(determinant_matrix_energy_alpha)*np.linalg.det(determinant_matrix_beta))
                print(Hamiltonian_SLbasis_alpha)
                print(adj_alpha)
                print(Hamiltonian_SLbasis_alpha@adj_alpha)
                sys.exit(1)
        return energy_1e
    def twobody_energy(self,determinant_matrix,eribasis,permutations_left,permutations_right):
        #return self.twobody_energy_alt(determinant_matrix,eribasis,permutations_left,permutations_right)
        threshold=1e-14
        Linv_a,da,Rinv_a=LDU_decomp(determinant_matrix[0])#,threshold=threshold) #Alpha LdR decomposition
        Linv_b,db,Rinv_b=LDU_decomp(determinant_matrix[1])#,threshold=threshold) #Beta LdR decomposition
        neh=self.number_electronshalf
        num_singularities_left=len(da[np.abs(da)<threshold])
        num_singularities_right=len(db[np.abs(db)<threshold])
        num_singularities=num_singularities_left+num_singularities_right
        if num_singularities>=3:
            return 0

        neh=self.number_electronshalf
        n=int(self.number_electronshalf*2)
        nh=self.number_electronshalf
        electrons_basis=np.arange(neh)
        electrons_alpha_left=np.where(electrons_basis==permutations_left[0][0],permutations_left[0][1],electrons_basis)
        electrons_beta_left=np.where(electrons_basis==permutations_left[1][0],permutations_left[1][1],electrons_basis)
        electrons_alpha_right=np.where(electrons_basis==permutations_right[0][0],permutations_right[0][1],electrons_basis)
        electrons_beta_right=np.where(electrons_basis==permutations_right[1][0],permutations_right[1][1],electrons_basis)
        eri_MO_aaaa=eribasis[np.ix_(electrons_alpha_left,electrons_alpha_right,electrons_alpha_left,electrons_alpha_right)]
        eri_MO_aabb=eribasis[np.ix_(electrons_alpha_left,electrons_alpha_right,electrons_beta_left,electrons_beta_right)]
        eri_MO_bbbb=eribasis[np.ix_(electrons_beta_left,electrons_beta_right,electrons_beta_left,electrons_beta_right)]
        energy_2e=0

        if num_singularities==2:
            eri_MO_bbaa=eribasis[np.ix_(electrons_beta_left,electrons_beta_right,electrons_alpha_left,electrons_alpha_right)]
            d=np.concatenate((da,db))
            La=Linv_a.T
            Lb=Linv_b.T
            Ra=Rinv_a.T
            Rb=Rinv_b.T
            if num_singularities_right==2:
                i_index=n-2
                j_index=n-1
                k_index=n-2
                l_index=n-1
            elif num_singularities_left==2:
                i_index=nh-2
                j_index=nh-1
                k_index=nh-2
                l_index=nh-1
            elif num_singularities_left==1 and num_singularities_right==1:
                i_index=nh-1
                j_index=n-1
                k_index=nh-1
                l_index=n-1

            """
            last_column_R2_1,last_column_R2_2,last_column_R2_3=second_order_compound_col_separated(Ra,Rb,l=l_index,k=k_index)
            last_row_L2_1,last_row_L2_2,last_row_L2_3=second_order_compound_row_separated(La,Lb,j=j_index,i=i_index)
            G1s,G2s,G3s=get_antisymm_element_separated(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,n)
            M1s=np.outer(last_column_R2_1,last_row_L2_1)
            M2s=np.outer(last_column_R2_2,last_row_L2_2)
            M3s=np.outer(last_column_R2_3,last_row_L2_3)
            energy_2e=np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))+np.trace(dot_nb(M3s,G3s))
            """
            L=scipy.linalg.block_diag(La,Lb)
            R=scipy.linalg.block_diag(Ra,Rb)
            last_column_R2=second_order_compound_col(R,l=l_index,k=k_index)
            last_row_L2=second_order_compound_row(L,j=j_index,i=i_index)
            nonzero_d=d[np.abs(d)>1e-10]
            d_val=np.prod(nonzero_d)
            adjugate_2S=np.outer(last_column_R2,last_row_L2)*d_val
            G=get_antisymm_element_full(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,eri_MO_bbaa,n)
            energy_2e=np.trace(dot_nb(adjugate_2S,G))
            #assert(np.abs(energy_2e-self.twobody_energy_alt(determinant_matrix,eribasis,permutations_left,permutations_right))<1e-10)
        elif num_singularities==1:
            if num_singularities_left==1: #Left matrix is "almost non-invertible"
                nonzero_row,nonzero_col=cofactor_index(determinant_matrix[0])
                determinant_0_new=determinant_matrix[0].copy()
                determinant_0_new[nonzero_row,nonzero_col]=determinant_0_new[nonzero_row,nonzero_col]+1
                M1_1,M2_1,M3_1=second_order_adj_matrix_blockdiag_separated(determinant_0_new,determinant_matrix[1])
                determinant_0_new[nonzero_row,nonzero_col]=determinant_0_new[nonzero_row,nonzero_col]-2
                M1_2,M2_2,M3_2=second_order_adj_matrix_blockdiag_separated(determinant_0_new,determinant_matrix[1])
                M1s=0.5*(M1_1+M1_2)
                M2s=0.5*(M2_1+M2_2)
                M3s=0.5*(M3_1+M3_2)
            elif num_singularities_right==1:
                nonzero_row,nonzero_col=cofactor_index(determinant_matrix[1])
                determinant_1_new=determinant_matrix[1].copy()
                determinant_1_new[nonzero_row,nonzero_col]=determinant_1_new[nonzero_row,nonzero_col]+1
                M1_1,M2_1,M3_1=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_1_new)
                determinant_1_new[nonzero_row,nonzero_col]=determinant_1_new[nonzero_row,nonzero_col]-2
                M1_2,M2_2,M3_2=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_1_new)
                M1s=0.5*(M1_1+M1_2)
                M2s=0.5*(M2_1+M2_2)
                M3s=0.5*(M3_1+M3_2)
            M1s=0.5*(M1_1+M1_2)
            M2s=0.5*(M2_1+M2_2)
            M3s=0.5*(M3_1+M3_2)
            G1s,G2s,G3s=get_antisymm_element_separated(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,n)
            energy_2e=np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))+np.trace(dot_nb(M3s,G3s))
        else:
            G1s,G2s,G3s=get_antisymm_element_separated(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,n)
            M1s,M2s,M3s=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_matrix[1])
            energy_2e=np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))+np.trace(dot_nb(M3s,G3s))

        return energy_2e
    def twobody_energy_alt(self,determinant_matrix,eribasis,permutations_left,permutations_right):
        neh=self.number_electronshalf
        electrons_basis=np.arange(neh)
        electrons_alpha_left=np.where(electrons_basis==permutations_left[0][0],permutations_left[0][1],electrons_basis)
        electrons_beta_left=np.where(electrons_basis==permutations_left[1][0],permutations_left[1][1],electrons_basis)
        electrons_alpha_right=np.where(electrons_basis==permutations_right[0][0],permutations_right[0][1],electrons_basis)
        electrons_beta_right=np.where(electrons_basis==permutations_right[1][0],permutations_right[1][1],electrons_basis)
        eri_MO_aaaa=eribasis[np.ix_(electrons_alpha_left,electrons_alpha_right,electrons_alpha_left,electrons_alpha_right)]
        eri_MO_aabb=eribasis[np.ix_(electrons_alpha_left,electrons_alpha_right,electrons_beta_left,electrons_beta_right)]
        eri_MO_bbbb=eribasis[np.ix_(electrons_beta_left,electrons_beta_right,electrons_beta_left,electrons_beta_right)]
        eri_MO_bbaa=eribasis[np.ix_(electrons_beta_left,electrons_beta_right,electrons_alpha_left,electrons_alpha_right)]
        energy_2e=0
        determinant_matrix_alpha=determinant_matrix[0]
        determinant_matrix_beta=determinant_matrix[1]
        number_electronshalf=self.number_electronshalf
        number_electrons_alpha=len(electrons_alpha_left)
        number_electrons_beta=len(electrons_beta_left)
        large_S=np.zeros((number_electronshalf*2,number_electronshalf*2))
        large_S[:number_electrons_alpha,:number_electrons_alpha]=determinant_matrix_alpha.copy()
        large_S[number_electrons_alpha:,number_electrons_alpha:]=determinant_matrix_beta.copy()
        for k in range(number_electronshalf*2):
            for l in range(k+1,number_electronshalf*2):
                largeS_2e=large_S.copy()
                largeS_2e[:,k]=0
                largeS_2e[:,l]=0
                for a in range(number_electronshalf*2):
                    for b in range(number_electronshalf*2):
                        largeS_2e[a,k]=1
                        largeS_2e[b,l]=1
                        largeS_2e[a-1,k]=0
                        largeS_2e[b-1,l]=0
                        if(k<number_electrons_alpha and l<number_electrons_alpha and a < number_electrons_alpha and b< number_electrons_alpha): #alpha, alpha
                            eri_of_interest=eri_MO_aaaa[a,k,b,l]
                        elif(k>=number_electrons_alpha and l>=number_electrons_alpha and a >= number_electrons_alpha and b>= number_electrons_alpha): #beta, beta
                            eri_of_interest=eri_MO_bbbb[a-number_electrons_alpha,k-number_electrons_alpha,b-number_electrons_alpha,l-number_electrons_alpha]
                        elif(k<number_electrons_alpha and l>=number_electrons_alpha and a < number_electrons_alpha and b>= number_electrons_alpha):#alpha, beta
                            eri_of_interest=eri_MO_aabb[a,k,b-number_electrons_alpha,l-number_electrons_alpha]
                        elif(k>=number_electrons_alpha and l<number_electrons_alpha and a >= number_electrons_alpha and b< number_electrons_alpha): #beta,alpha and b>= number_electronshalf):#alpha, beta
                                eri_of_interest=eri_MO_bbaa[a-number_electrons_alpha,k-number_electrons_alpha,b,l]
                        else:
                            continue
                        if(abs(eri_of_interest)>=1e-10):
                            det=np.linalg.det(largeS_2e)
                            if(det>1e-10):
                                energy_2e+=eri_of_interest*det
        return energy_2e

class eigensolver_RHF_knowncoefficients(eigvecsolver_RHF):
    def __init__(self,sample_coefficients,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False):
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
    def calculate_energies(self,xc_array):
        """Calculates the molecule's energy"""
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.HF_coefficients)):
                new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf])
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            try:
                eigval,eigvec=generalized_eigenvector(T,S)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval
            eigval_array.append(eigval)
        return energy_array,eigval_array
class eigvecsolver_UHF(eigvecsolver_RHF):
    def solve_HF(self):
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(x)
            mf=scf.UHF(mol)
            dm_alpha, dm_beta = mf.get_init_guess()
            dm_beta+=np.random.random_sample(dm_beta.shape)
            dm_beta[:2,:2] = 0
            dm = (dm_alpha,dm_beta)
            energy=mf.kernel(dm)
            expansion_coefficients_mol_alpha=mf.mo_coeff[0][:, mf.mo_occ[0] > 0.]
            expansion_coefficients_mol_beta =mf.mo_coeff[1][:, mf.mo_occ[1] > 0.]
            HF_coefficients.append([expansion_coefficients_mol_alpha,expansion_coefficients_mol_beta])
        self.HF_coefficients=HF_coefficients
    def calculate_energies(self,xc_array):
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.sample_points)):
                alpha=self.basischange(self.HF_coefficients[i][0],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf]
                beta =self.basischange(self.HF_coefficients[i][1],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf]
                new_HF_coefficients.append([alpha,beta])
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            eigval,eigvec=generalized_eigenvector(T,S)
            energy_array[index]=eigval
            eigval_array.append(eigval)
        return energy_array,eigval_array
    def getdeterminant_matrix(self,AO_overlap,HF_coefficients_left,HF_coefficients_right):
        determinant_matrix_alpha=np.einsum("ab,ai,bj->ij",AO_overlap,HF_coefficients_left[0],HF_coefficients_right[0])
        determinant_matrix_beta=np.einsum("ab,ai,bj->ij",AO_overlap,HF_coefficients_left[1],HF_coefficients_right[1])
        return [determinant_matrix_alpha,determinant_matrix_beta]

    def getoverlap(self,determinant_matrix):
        overlap=np.linalg.det(determinant_matrix[0])*np.linalg.det(determinant_matrix[1]) #alpha part times beta part
        return overlap

    def onebody_energy(self,energy_basis_1e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
        number_electronshalf=self.number_electronshalf
        Hamiltonian_SLbasis_alpha=np.einsum("ki,lj,kl->ij",HF_coefficients_left[0],HF_coefficients_right[0],energy_basis_1e) #Hamilton operator in Slater determinant basis
        Hamiltonian_SLbasis_beta=np.einsum("ki,lj,kl->ij",HF_coefficients_left[1],HF_coefficients_right[1],energy_basis_1e) #Hamilton operator in Slater determinant basis
        energy_1e=0
        determinant_matrix_alpha=determinant_matrix[0]
        determinant_matrix_beta=determinant_matrix[1]
        for k in range(number_electronshalf):
            determinant_matrix_energy_alpha=determinant_matrix_alpha.copy() #Re-initiate Energy matrix
            for l in range(number_electronshalf):
                    determinant_matrix_energy_alpha[l,k]=Hamiltonian_SLbasis_alpha[l,k]
            energy_contribution=np.linalg.det(determinant_matrix_energy_alpha)*np.linalg.det(determinant_matrix_beta)
            energy_1e+=energy_contribution
        for k in range(number_electronshalf):
            determinant_matrix_energy_beta=determinant_matrix_beta.copy() #Re-initiate Energy matrix
            for l in range(number_electronshalf):
                    determinant_matrix_energy_beta[l,k]=Hamiltonian_SLbasis_beta[l,k]
            energy_contribution=np.linalg.det(determinant_matrix_energy_beta)*np.linalg.det(determinant_matrix_alpha)
            energy_1e+=energy_contribution
        return energy_1e
    def twobody_energy(self,energy_basis_2e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
        eri_MO_aabb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
        eri_MO_aaaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_bbbb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")

        energy_2e=0
        n=int(self.number_electronshalf*2)
        nh=self.number_electronshalf
        G1s,G2s,G3s=get_antisymm_element_separated(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,int(n))
        M1s,M2s,M3s=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_matrix[1])
        energy_2e=np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))+np.trace(dot_nb(M3s,G3s))
        return energy_2e

class eigvecsolver_UHF_singles(eigvecsolver_UHF):
    def solve_HF(self):
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(x)
            mf=scf.UHF(mol)
            dm_alpha, dm_beta = mf.get_init_guess()
            dm_beta+=np.random.random_sample(dm_beta.shape)
            dm_beta[:2,:2] = 0
            dm = (dm_alpha,dm_beta)
            energy=mf.kernel(dm)
            expansion_coefficients_mol_alpha=mf.mo_coeff[0][:, mf.mo_occ[0] >= 0.]
            expansion_coefficients_mol_beta =mf.mo_coeff[1][:, mf.mo_occ[1] >= 0.]
            HF_coefficients.append([expansion_coefficients_mol_alpha,expansion_coefficients_mol_beta])
        self.HF_coefficients=HF_coefficients
    def calculate_energies(self,xc_array):
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_basis=[]
            for i in range(len(self.sample_points)):
                alpha=self.basischange(self.HF_coefficients[i][0],mol_xc.intor("int1e_ovlp"))
                beta=self.basischange(self.HF_coefficients[i][1],mol_xc.intor("int1e_ovlp"))
                new_HF_basis.append([alpha,beta]) #Now I have a basis for alphas and betas.
            #First step: Create singles. That menas that each alpha can be made to a beta, and each beta to an alpha.
            single_basis=[] #single_basis[i] - "zero"+singles created at point x_i. single_basis[i][j] -> The j'th singly excited determinant. single_basis[i][j][0] - it's alpha state.
            for ai,bi in new_HF_basis:
                all_determinants_i=self.create_all_determinants(ai,bi)
                single_basis.append(all_determinants_i)
            S,T=self.calculate_ST_matrices(mol_xc,single_basis)
            eigval,eigvec=generalized_eigenvector(T,S)
            energy_array[index]=eigval
            eigval_array.append(eigval)
        return energy_array,eigval_array
    def create_singles(self,alpha,beta):
        basisset_size=len(alpha[:,0])
        n_occ=self.number_electronshalf
        n_unocc=basisset_size-n_occ
        alpha_alpha_permutations=[]
        #1. Create all possible alpha-to-alpha
        for i in range(n_occ):
            for j in range(n_occ,basisset_size):
                alpha_alpha_permutations.append(np.array([i,j])) #This means: i out, j in!
        #2. Create all possible beta-to-beta
        beta_beta_permutations=alpha_alpha_permutations.copy() #Exactly the same
        #3. Create all possible alpha-to-beta and beta-to-alpha
        alpha_beta_permutations=alpha_alpha_permutations.copy()
        beta_alpha_permutations=alpha_alpha_permutations.copy()
        return alpha_alpha_permutations,beta_beta_permutations,alpha_beta_permutations,beta_alpha_permutations
    def create_all_determinants(self,alpha,beta):
        aa,bb,ab,ba=self.create_singles(alpha,beta)
        HF_states=[]
        HF_states.append([alpha[:,:self.number_electronshalf],beta[:,:self.number_electronshalf]])

        for i,j in aa:
            state=alpha.copy()
            state[:,[i, j]] = state[:,[j, i]] #swap i and j
            HF_states.append([state[:,:self.number_electronshalf],beta[:,:self.number_electronshalf]])
        for i,j in bb:
            state=beta.copy()
            state[:,[i, j]] = state[:,[j, i]] #swap i and j
            HF_states.append([alpha[:,:self.number_electronshalf],state[:,:self.number_electronshalf]])
        for i,j in ab:
            state_a=alpha.copy()[:,:self.number_electronshalf]
            state_b=beta.copy()[:,:self.number_electronshalf]
            state_a=np.delete(state_a,i,1)#Delete column
            state_b=np.c_[state_b,beta[:,j]] #Add extra column to beta
            HF_states.append([state_a,state_b])
        for i,j in ba:
            state_b=beta.copy()[:,:self.number_electronshalf]
            state_a=alpha.copy()[:,:self.number_electronshalf]
            state_b=np.delete(state_b,i,1)#Delete column
            state_a=np.c_[state_a,alpha[:,j]] #Add extra column to beta
            HF_states.append([state_a,state_b])
        return HF_states
    def calculate_ST_matrices(self,mol_xc,single_basis):
        number_of_points=len(single_basis)
        number_of_determinants_per_point=len(single_basis[0])
        self.num_det_total=number_of_determinants_per_point*number_of_points
        number_electronshalf=self.number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        S=np.zeros((self.num_det_total,self.num_det_total))
        T=np.zeros((self.num_det_total,self.num_det_total))
        for i in range(self.num_det_total):
            for j in range(i,self.num_det_total):
                sampling_point_left=i//number_of_determinants_per_point
                sampling_determinant_left=(i-number_of_determinants_per_point*sampling_point_left)
                sampling_point_right=j//number_of_determinants_per_point
                sampling_determinant_right=(j-number_of_determinants_per_point*sampling_point_right)
                if sampling_point_left==sampling_point_right:
                    #Apply standard slater-Condon-Rules
                    pass
                else:
                    pass
                Det_L=single_basis[sampling_point_left][sampling_determinant_left]
                Det_R=single_basis[sampling_point_right][sampling_determinant_right]
                Det_L_numalpha=len(Det_L[0][0,:])
                Det_R_numalpha=len(Det_R[0][0,:])
                if(Det_L_numalpha!=Det_R_numalpha):
                    continue #No cupling whatsoever!!

                determinant_matrix=self.getdeterminant_matrix(overlap_basis,Det_L,Det_R)
                overlap=self.getoverlap(determinant_matrix)
                S[i,j]=S[j,i]=overlap
                nuc_repulsion_energy=mol_xc.energy_nuc()*overlap
                energy_1e=self.onebody_energy(energy_basis_1e,Det_L,Det_R,determinant_matrix)
                energy_2e=self.twobody_energy(energy_basis_2e,Det_L,Det_R,determinant_matrix)
                energy_total=energy_2e+energy_1e+nuc_repulsion_energy
                T[i,j]=energy_total
                T[j,i]=energy_total
        return S,T

    def onebody_energy(self,energy_basis_1e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
        number_electrons_alpha=len(HF_coefficients_left[0][0,:])
        number_electrons_beta=len(HF_coefficients_left[1][0,:])
        Hamiltonian_SLbasis_alpha=np.einsum("ki,lj,kl->ij",HF_coefficients_left[0],HF_coefficients_right[0],energy_basis_1e) #Hamilton operator in Slater determinant basis
        Hamiltonian_SLbasis_beta=np.einsum("ki,lj,kl->ij",HF_coefficients_left[1],HF_coefficients_right[1],energy_basis_1e) #Hamilton operator in Slater determinant basis
        energy_1e=0
        determinant_matrix_alpha=determinant_matrix[0]
        determinant_matrix_beta=determinant_matrix[1]
        for k in range(number_electrons_alpha):
            determinant_matrix_energy_alpha=determinant_matrix_alpha.copy() #Re-initiate Energy matrix
            for l in range(number_electrons_alpha):
                    determinant_matrix_energy_alpha[l,k]=Hamiltonian_SLbasis_alpha[l,k]
            energy_contribution=np.linalg.det(determinant_matrix_energy_alpha)*np.linalg.det(determinant_matrix_beta)
            energy_1e+=energy_contribution
        for k in range(number_electrons_beta):
            determinant_matrix_energy_beta=determinant_matrix_beta.copy() #Re-initiate Energy matrix
            for l in range(number_electrons_beta):
                    determinant_matrix_energy_beta[l,k]=Hamiltonian_SLbasis_beta[l,k]
            energy_contribution=np.linalg.det(determinant_matrix_energy_beta)*np.linalg.det(determinant_matrix_alpha)
            energy_1e+=energy_contribution
        return energy_1e
    def twobody_energy(self,energy_basis_2e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
        eri_MO_aabb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
        eri_MO_aaaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_bbbb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
        determinant_matrix_alpha=determinant_matrix[0]
        determinant_matrix_beta=determinant_matrix[1]
        n=int(self.number_electronshalf*2)
        number_electrons_alpha=len(HF_coefficients_left[0][0,:])
        number_electrons_beta=len(HF_coefficients_left[1][0,:])
        G1s,G2s,G3s=get_antisymm_element_separated(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,int(n),number_electrons_alpha,number_electrons_beta)
        M1s,M2s,M3s=second_order_adj_matrix_blockdiag_separated(determinant_matrix_alpha.copy(),determinant_matrix_beta.copy())
        energy_2e=np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))+np.trace(dot_nb(M3s,G3s))
        return energy_2e
    def twobody_energy_old(self,energy_basis_2e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
        eri_MO_aabb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
        eri_MO_bbaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_aaaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_bbbb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
        energy_2e=0
        determinant_matrix_alpha=determinant_matrix[0]
        determinant_matrix_beta=determinant_matrix[1]
        number_electronshalf=self.number_electronshalf
        number_electrons_alpha=len(HF_coefficients_left[0][0,:])
        number_electrons_beta=len(HF_coefficients_left[1][0,:])
        large_S=np.zeros((number_electronshalf*2,number_electronshalf*2))
        large_S[:number_electrons_alpha,:number_electrons_alpha]=determinant_matrix_alpha.copy()
        large_S[number_electrons_alpha:,number_electrons_alpha:]=determinant_matrix_beta.copy()
        for k in range(number_electronshalf*2):
            for l in range(k+1,number_electronshalf*2):
                largeS_2e=large_S.copy()
                largeS_2e[:,k]=0
                largeS_2e[:,l]=0
                for a in range(number_electronshalf*2):
                    for b in range(number_electronshalf*2):
                        largeS_2e[a,k]=1
                        largeS_2e[b,l]=1
                        largeS_2e[a-1,k]=0
                        largeS_2e[b-1,l]=0
                        if(k<number_electrons_alpha and l<number_electrons_alpha and a < number_electrons_alpha and b< number_electrons_alpha): #alpha, alpha
                            eri_of_interest=eri_MO_aaaa[a,k,b,l]
                        elif(k>=number_electrons_alpha and l>=number_electrons_alpha and a >= number_electrons_alpha and b>= number_electrons_alpha): #beta, beta
                            eri_of_interest=eri_MO_bbbb[a-number_electrons_alpha,k-number_electrons_alpha,b-number_electrons_alpha,l-number_electrons_alpha]
                        elif(k<number_electrons_alpha and l>=number_electrons_alpha and a < number_electrons_alpha and b>= number_electrons_alpha):#alpha, beta
                            eri_of_interest=eri_MO_aabb[a,k,b-number_electrons_alpha,l-number_electrons_alpha]
                        elif(k>=number_electrons_alpha and l<number_electrons_alpha and a >= number_electrons_alpha and b< number_electrons_alpha): #beta,alpha and b>= number_electronshalf):#alpha, beta
                                eri_of_interest=eri_MO_bbaa[a-number_electrons_alpha,k-number_electrons_alpha,b,l]
                        else:
                            continue
                        if(abs(eri_of_interest)>=1e-10):
                            energy_2e+=np.linalg.det(largeS_2e)*eri_of_interest
        return energy_2e

def energy_curve_UHF(xvals,basis_type,molecule):
    energies=[]
    for x in xvals:
        mol1=gto.Mole()
        mol1.atom=molecule(x) #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit='AU'
        mol1.spin=0 #Assume closed shell
        mol1.verbose=2
        mol1.build()
        mf=scf.UHF(mol1)
        dm_alpha, dm_beta = mf.get_init_guess()
        dm_beta+=np.random.random_sample(dm_beta.shape)
        dm_beta[:2,:2] = 0
        dm = (dm_alpha,dm_beta)

        energy=mf.kernel(dm)
        energies.append(energy)
    return np.array(energies)
def energy_curve_RHF(xvals,basis_type,molecule):
    energies=[]
    for x in xvals:
        mol1=gto.Mole()
        mol1.atom=molecule(x) #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit='AU'
        mol1.spin=0 #Assume closed shell
        mol1.verbose=2
        mol1.symmetry=True
        mol1.build()
        mf=scf.RHF(mol1)
        energy=mf.kernel()
        energies.append(energy)
    return np.array(energies)

def CC_energy_curve(xvals,basis_type,molecule):
    energies=[]
    for index,x in enumerate(xvals):
        print("%d/%d"%(index,len(xvals)))
        mol1=gto.Mole()
        mol1.atom=molecule(x) #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit='AU'
        mol1.spin=0 #Assume closed shell
        mol1.symmetry=True
        mol1.build()
        mf=mol1.RHF().run(verbose=2) #Solve RHF equations to get overlap
        ccsolver=cc.CCSD(mf).run(verbose=2)
        energy=ccsolver.e_tot
        energy+= ccsolver.ccsd_t()
        energies.append(energy)
    return np.array(energies)
def FCI_energy_curve(xvals,basis_type,molecule):
    energies=[]
    for index,x in enumerate(xvals):
        print("%d/%d"%(index,len(xvals)))
        mol1=gto.Mole()
        mol1.atom=molecule(x) #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit='AU'
        mol1.spin=0 #Assume closed shell
        mol1.symmetry=True
        mol1.build()
        mf=mol1.RHF().run(verbose=2) #Solve RHF equations to get overlap
        cisolver = fci.FCI(mol1, mf.mo_coeff)
        e, fcivec = cisolver.kernel()
        energies.append(e)
    return np.array(energies)
def CASCI_energy_curve(xvals,basis_type,molecule):
    energies=[]
    mol1=gto.Mole()
    mol1.atom=molecule(x) #take this as a "basis" assumption.
    mol1.basis=basis_type
    mol1.unit='AU'
    mol1.symmetry=True
    mol1.spin=0 #Assume closed shell
    myhf = mol.RHF().run()
    # Use MP2 natural orbitals to define the active space for the single-point CAS-CI calculation
    mymp = mp.UMP2(myhf).run()

    noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
    ncas, nelecas = (6,8)
    mycas = mcscf.CASCI(myhf, ncas, nelecas)
    energyies.append(mycas.kernel(natorbs))
if __name__=="__main__":
    from timeit import default_timer as timer

    plt.figure(figsize=(9,6))

    basis="6-31G"
    sample_x=np.linspace(1.5,2.0,11)
    xc_array=np.linspace(1.2,5.0,20)
    molecule=lambda x: """F 0 0 0; H 0 0 %f"""%x
    molecule_name=r"Hydrogen Fluoride"
    '''
    basis="6-31G*"
    molecule_name="BeH2"
    sample_x=np.linspace(0,2.5,151)
    xc_array=np.linspace(3,4,11)
    molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
    '''
    print("FCI")
    #energiesFCI=FCI_energy_curve(xc_array,basis,molecule=molecule)
    print("CCSDT")
    energiesCC=CC_energy_curve(xc_array,basis,molecule=molecule)
    start=timer()
    for i in range(1,len(sample_x)+1,2):
        print("Eigvec (%d)"%(i))
        HF=eigvecsolver_RHF(sample_x[:i],basis,molecule=molecule,symmetry="C2v")
        energiesEC,eigenvectors=HF.calculate_energies(xc_array)
        plt.plot(xc_array,energiesEC,label="EC (%d point(s)), %s"%(i,basis))
    end=timer()
    print("Time elapsed: %f"%(end-start))
    #In addition:"outliers":
    #HF=eigvecsolver_RHF([2.5,3],basis,molecule=molecule)
    #energiesEC,eigenvectors=HF.calculate_energies(xc_array)
    #plt.plot(xc_array,energiesEC,label="EC (2.5 and 3)")
    #print("UHF")

    energiesHF=energy_curve_RHF(xc_array,basis,molecule=molecule)
    energiesHF_sample=energy_curve_RHF(sample_x,basis,molecule=molecule)
    sample_x=sample_x
    xc_array=xc_array
    plt.plot(xc_array,energiesHF,label="RHF,%s"%basis)
    plt.plot(xc_array,energiesCC,label="CCSD(T),%s"%basis)
    #plt.plot(xc_array,energiesFCI,label="FCI,%s"%basis)
    #plt.plot(sample_x,energiesHF_sample,"o",color="black",label="Sample points")
    plt.xlabel("Atomic distance (Bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.title("%s potential energy curve"%molecule_name)
    plt.legend(loc="lower right")
    plt.tight_layout()
    #plt.ylim([-100.3,-99.4])
    plt.savefig("EC_RHF_%s.pdf"%molecule_name)
    plt.show()
    plt.plot(xc_array,energiesEC-energiesFCI,label="EC (max)-FCI")
    plt.plot(xc_array,energiesHF-energiesFCI,label="RHF-FCI")
    plt.legend()
    plt.show()
