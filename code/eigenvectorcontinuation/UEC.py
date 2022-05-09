from REC import *
import numpy as np
from pyscf import gto, scf, fci,cc,ao2mo, mp, mcscf
import pyscf
import sys
import scipy
from matrix_operations import *
from helper_functions import *
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
class eigvecsolver_UHF_coupling(eigvecsolver_UHF):
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
            mf = scf.UHF(mol)
            eri=mol.intor('int2e',aosym="s1")*x
            mf._eri = ao2mo.restore(1,eri,mol.nao_nr())
            mol.incore_anyway=True
            dm_alpha, dm_beta = mf.get_init_guess()
            dm_beta[:,:] = 0
            dm = (dm_alpha,dm_beta)
            mf.kernel(dm)
            expansion_coefficients_mol_alpha=mf.mo_coeff[0][:, mf.mo_occ[0] > 0.]
            expansion_coefficients_mol_beta =mf.mo_coeff[1][:, mf.mo_occ[1] > 0.]
            HF_coefficients.append([expansion_coefficients_mol_alpha,expansion_coefficients_mol_beta])
        self.HF_coefficients=HF_coefficients
    def calculate_energies(self,xc_array):
            """Calculates the molecule's energy"""
            energy_array=np.zeros(len(xc_array))
            eigval_array=[]
            for index,xc in enumerate(xc_array):
                self.solve_HF(xc) #Update HF coefficients
                mol_xc=self.build_molecule(xc)
                new_HF_coefficients=self.HF_coefficients #No need to basis change (same basis)
                S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
                try:
                    eigval,eigvec=generalized_eigenvector(T,S)
                except:
                    eigval=float('NaN')
                    eigvec=float('NaN')
                energy_array[index]=eigval
                eigval_array.append(eigval)
            return energy_array,eigval_array
