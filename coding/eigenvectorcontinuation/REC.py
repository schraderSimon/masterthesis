import numpy as np
from pyscf import gto, scf, fci,cc,ao2mo, mp, mcscf
import pyscf
import sys
import scipy
from matrix_operations import *
from helper_functions import *
np.set_printoptions(linewidth=200,precision=2,suppress=True)
import matplotlib
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
        eigvec_array=[]
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
            eigvec_array.append(eigvec)
        return energy_array,eigvec_array
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
                conv_sol=self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf]
                determinant_matrix=self.getdeterminant_matrix(overlap_basis,true_HF_coeffmatrix,conv_sol)
                overlap=self.getoverlap(determinant_matrix)
                overlaps[i]=overlap
            overlap_to_HF[index]=np.dot(overlaps,eigvecs[index])
        return overlap_to_HF
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
        self.all_new_HF_coefficients=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.sample_points)):
                new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:]) #Actually take the WHOLE thing.
                #new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf])
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
    def calculate_overlap_to_HF(self,xc_array):
        energies,eigvecs=self.calculate_energies(xc_array) #get energies and eigenvalues
        overlap_to_HF=np.zeros(len(energies))
        neh=self.number_electronshalf
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            "Calculate RHF solution"
            mol_xc=self.build_molecule(xc)
            overlap_basis=mol_xc.intor("int1e_ovlp")
            mf=mol_xc.RHF().run(verbose=2,conv_tol=1e-8)
            true_HF_coeffmatrix= mf.mo_coeff[:, mf.mo_occ >=0.5]
            "re-calculate HF coefficients"
            permutations=self.create_singles(self.all_new_HF_coefficients[index])
            number_matrix_elements=len(self.HF_coefficients)*(len(permutations))
            overlaps=np.zeros(number_matrix_elements)
            len_permutations=len(permutations)

            for c1 in range(number_matrix_elements):

                i=c1//len_permutations
                j=c1%len_permutations
                GS_bra=true_HF_coeffmatrix
                GS_ket=self.all_new_HF_coefficients[index][i][:,:neh]
                ES_ket=swap_cols(self.all_new_HF_coefficients[index][i],permutations[j][0],permutations[j][1])[:,:neh]
                determinant_matrix_GS_GS=self.getdeterminant_matrix(overlap_basis,GS_bra,GS_ket) #The unmodified part
                determinant_matrix_GS_EX=self.getdeterminant_matrix(overlap_basis,GS_bra,ES_ket)
                if j==0:
                    determinant_matrix=self.getdeterminant_matrix(overlap_basis,true_HF_coeffmatrix,self.all_new_HF_coefficients[index][i][:,:self.number_electronshalf])
                    overlap=self.getoverlap([determinant_matrix,determinant_matrix])
                    overlaps[c1]=overlap
                    if(abs(overlap<1e-12)):
                        overlaps[c1]=0
                else:
                    multiplier=1/np.sqrt(2)
                    determinant_matrix_1=[determinant_matrix_GS_GS,determinant_matrix_GS_EX] #<GS,GS|GS,ex>
                    determinant_matrix_2=[determinant_matrix_GS_EX,determinant_matrix_GS_GS] #<GS,GS|ex,GS>
                    overlap=(self.getoverlap(determinant_matrix_1)+self.getoverlap(determinant_matrix_2))*multiplier
                    if(abs(overlap)<1e-12):
                        overlap=0
                    overlaps[c1]=overlap
            #print("Overlaps")

            #print(np.around(np.array(list(zip(overlaps,eigvecs[index]))),2))
            #print("Eigvecs")
            #print(eigvecs[index])
            #print(abs(overlaps).T@abs(eigvecs[index]))
            overlap_to_HF[index]=overlaps.T@eigvecs[index]
        return overlap_to_HF, energies,eigvecs

    def getoverlap(self,determinant_matrix):
        overlap=np.linalg.det(determinant_matrix[0])*np.linalg.det(determinant_matrix[1]) #alpha part times beta part
        return overlap

    def calculate_ST_matrices(self,mol_xc,n_HF_coef):
        number_electronshalf=self.number_electronshalf
        neh=number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        permutations=self.create_singles(n_HF_coef)
        number_matrix_elements=len(n_HF_coef)*(len(permutations))
        S=np.zeros((number_matrix_elements,number_matrix_elements))
        T=np.zeros((number_matrix_elements,number_matrix_elements))
        e1_slbasis=[]
        e2_slbasis=[]
        for i in range(len(self.HF_coefficients)): #Calculate e1-basis and e2-basis for all combinations of determinants
            temp_1=[]
            temp_2=[]
            for k in range(len(self.HF_coefficients)):
                temp_1.append(np.einsum("ki,lj,kl->ij",n_HF_coef[i],n_HF_coef[k],energy_basis_1e))
                basis=ao2mo.get_mo_eri(energy_basis_2e,(n_HF_coef[i],n_HF_coef[k],n_HF_coef[i],n_HF_coef[k]),aosym="s1")
                temp_2.append(basis)
            e1_slbasis.append(temp_1)
            e2_slbasis.append(temp_2)
        len_permutations=len(permutations)
        for c1 in range(number_matrix_elements): #For each matrix element
            print("%d %d"%(c1,number_matrix_elements))
            i=c1//len_permutations # The number of the HF determinant
            j=c1%len_permutations #Number of the excitation
            for c2 in range(c1,number_matrix_elements):
                k=c2//len_permutations
                l=c2%len_permutations
                energy_1e=0
                energy_2e=0
                if l==0 and j==0: #Both are non-excited states
                    determinant=self.getdeterminant_matrix(overlap_basis,n_HF_coef[i][:,:neh],n_HF_coef[k][:,:neh])
                    determinant_alpha=determinant
                    determinant_beta=determinant
                    overlap=self.getoverlap([determinant_alpha,determinant_beta]) #Alpha and beta parts of S-matrix are identical.
                    p_bra_a,p_bra_b,p_ket_a,p_ket_b=self.getpermutation_s(permutations,0,0,0,0) #No permutations are performed, this is to work with the functions
                    energy_1e=self.onebody_energy([determinant,determinant],e1_slbasis[i][k],p_bra_a,p_bra_b,p_ket_a,p_ket_b)
                    energy_2e=self.twobody_energy([determinant,determinant],e2_slbasis[i][k],p_bra_a,p_bra_b,p_ket_a,p_ket_b)

                    S[c1,c2]=S[c2,c1]=overlap
                    #Calculate energy of CAS.
                    nuc_repulsion_energy=mol_xc.energy_nuc()*overlap
                    energy_total=energy_2e+energy_1e+nuc_repulsion_energy
                    T[c1,c2]=T[c2,c1]=energy_total
                    continue
                GS_bra=n_HF_coef[i][:,:neh]
                GS_ket=n_HF_coef[k][:,:neh]
                ES_bra=swap_cols(n_HF_coef[i],permutations[j][0],permutations[j][1])[:,:neh]
                ES_ket=swap_cols(n_HF_coef[k],permutations[l][0],permutations[l][1])[:,:neh]
                determinant_matrix_GS_GS=self.getdeterminant_matrix(overlap_basis,GS_bra,GS_ket) #The unmodified part
                determinant_matrix_EX_EX=self.getdeterminant_matrix(overlap_basis,ES_bra,ES_ket)
                determinant_matrix_GS_EX=self.getdeterminant_matrix(overlap_basis,GS_bra,ES_ket)
                determinant_matrix_EX_GS=self.getdeterminant_matrix(overlap_basis,ES_bra,GS_ket)
                if(j==0 or l==0):
                    multiplier=1/np.sqrt(2)
                    if j==0: #Bra is ground state
                        determinant_matrix_1=[determinant_matrix_GS_GS,determinant_matrix_GS_EX] #<GS,GS|GS,ex>
                        determinant_matrix_2=[determinant_matrix_GS_EX,determinant_matrix_GS_GS] #<GS,GS|ex,GS>
                        first_permutation=self.getpermutation_s(permutations,0,0,0,l)
                        second_permutation=self.getpermutation_s(permutations,0,0,l,0)
                        overlap=(self.getoverlap(determinant_matrix_1)+self.getoverlap(determinant_matrix_2))*multiplier
                        if(np.abs(overlap)<1e-12):
                            overlap=0
                        S[c2,c1]=S[c1,c2]=overlap
                        nuc_repulsion_energy=mol_xc.energy_nuc()*overlap/multiplier
                        energy_1e+=self.onebody_energy(determinant_matrix_1,e1_slbasis[i][k],*first_permutation)
                        energy_2e+=self.twobody_energy(determinant_matrix_1,e2_slbasis[i][k],*first_permutation)
                        energy_1e+=self.onebody_energy(determinant_matrix_2,e1_slbasis[i][k],*second_permutation)
                        energy_2e+=self.twobody_energy(determinant_matrix_2,e2_slbasis[i][k],*second_permutation)
                        energy_total=(energy_2e+energy_1e+nuc_repulsion_energy)*multiplier
                        T[c1,c2]=T[c2,c1]=energy_total
                    if l==0: #Ket is ground state
                        determinant_matrix_1=[determinant_matrix_GS_GS,determinant_matrix_EX_GS] #<GS,ex|GS,GS>
                        determinant_matrix_2=[determinant_matrix_EX_GS,determinant_matrix_GS_GS] #<ex,GS|GS,GS>
                        first_permutation=self.getpermutation_s(permutations,0,j,0,0)
                        second_permutation=self.getpermutation_s(permutations,j,0,0,0)
                        overlap=(self.getoverlap(determinant_matrix_1)+self.getoverlap(determinant_matrix_2))*multiplier
                        S[c2,c1]=S[c1,c2]=overlap
                        nuc_repulsion_energy=mol_xc.energy_nuc()*overlap/multiplier
                        energy_1e+=self.onebody_energy(determinant_matrix_1,e1_slbasis[i][k],*first_permutation)
                        energy_2e+=self.twobody_energy(determinant_matrix_1,e2_slbasis[i][k],*first_permutation)
                        energy_1e+=self.onebody_energy(determinant_matrix_2,e1_slbasis[i][k],*second_permutation)
                        energy_2e+=self.twobody_energy(determinant_matrix_2,e2_slbasis[i][k],*second_permutation)
                        energy_total=(energy_2e+energy_1e+nuc_repulsion_energy)*multiplier
                        T[c1,c2]=T[c2,c1]=energy_total#=T[k*(len(permutations))+l,c1]=energy_total
                    continue
                else:
                    multiplier=1 #*0.5
                    determinant_matrix_1=[determinant_matrix_GS_GS,determinant_matrix_EX_EX] #<GS,ex|GS,ex>
                    determinant_matrix_2=[determinant_matrix_GS_EX,determinant_matrix_EX_GS] #<GS,ex|ex,GS>
                    #determinant_matrix_3=[determinant_matrix_EX_GS,determinant_matrix_GS_EX] #<ex,GS|GS,ex>
                    #determinant_matrix_4=[determinant_matrix_EX_EX,determinant_matrix_GS_GS] #<ex,GS|ex,GS>
                    first_permutation=self.getpermutation_s(permutations,0,j,0,l)
                    second_permutation=self.getpermutation_s(permutations,0,j,l,0)
                    #third_permutation=self.getpermutation_s(permutations,j,0,0,l)
                    #fourth_permutation=self.getpermutation_s(permutations,j,0,l,0)
                    overlap=(self.getoverlap(determinant_matrix_1)+self.getoverlap(determinant_matrix_2))*multiplier
                    #overlap+=(self.getoverlap(determinant_matrix_3)+self.getoverlap(determinant_matrix_4))*multiplier
                    if(np.abs(overlap)<1e-12):
                        overlap=0
                    S[c2,c1]=S[c1,c2]=overlap
                    nuc_repulsion_energy=mol_xc.energy_nuc()*overlap/multiplier
                    energy_1e+=self.onebody_energy(determinant_matrix_1,e1_slbasis[i][k],*first_permutation)
                    energy_2e+=self.twobody_energy(determinant_matrix_1,e2_slbasis[i][k],*first_permutation)
                    energy_1e+=self.onebody_energy(determinant_matrix_2,e1_slbasis[i][k],*second_permutation)
                    energy_2e+=self.twobody_energy(determinant_matrix_2,e2_slbasis[i][k],*second_permutation)
                    #energy_1e+=self.onebody_energy(determinant_matrix_3,e1_slbasis[i][k],*third_permutation)
                    #energy_2e+=self.twobody_energy(determinant_matrix_3,e2_slbasis[i][k],*third_permutation)
                    #energy_1e+=self.onebody_energy(determinant_matrix_4,e1_slbasis[i][k],*fourth_permutation)
                    #energy_2e+=self.twobody_energy(determinant_matrix_4,e2_slbasis[i][k],*fourth_permutation)
                    energy_total=(energy_2e+energy_1e+nuc_repulsion_energy)*multiplier
                    T[c1,c2]=T[c2,c1]=energy_total#=T[k*(len(permutations))+l,c1]=energy_total
                    continue
        print("Done one")
        print(S)
        print(T)
        return S,T
    def getpermutation_s(self,permutations,bra_alpha,bra_beta,ket_alpha,ket_beta):
        permutations_bra_alpha=permutations[bra_alpha]
        permutations_bra_beta=permutations[bra_beta]
        permutations_ket_alpha=permutations[ket_alpha]
        permutations_ket_beta=permutations[ket_beta]
        return permutations_bra_alpha,permutations_bra_beta,permutations_ket_alpha,permutations_ket_beta
    def onebody_energy_alt(self,determinant_matrix,SLbasis,permutations_bra_a,permutations_bra_b,permutations_ket_a,permutations_ket_b):
        """Unlike the previous method, the left and right coefficients are now the WHOLE set (occ+unocc). The SLbasis is also the whole (for the corresponding system). The permutations contain all information
        about the construction of the coefficients.
        """

        neh=self.number_electronshalf
        electrons_basis=np.arange(neh)
        electrons_alpha_bra=np.where(electrons_basis==permutations_bra_a[0],permutations_bra_a[1],electrons_basis)
        electrons_beta_bra=np.where(electrons_basis==permutations_bra_b[0],permutations_bra_b[1],electrons_basis)
        electrons_alpha_ket=np.where(electrons_basis==permutations_ket_a[0],permutations_ket_a[1],electrons_basis)
        electrons_beta_ket=np.where(electrons_basis==permutations_ket_b[0],permutations_ket_b[1],electrons_basis)
        Hamiltonian_SLbasis_alpha=SLbasis[np.ix_(electrons_alpha_bra,electrons_alpha_ket)]
        Hamiltonian_SLbasis_beta=SLbasis[np.ix_(electrons_beta_bra,electrons_beta_ket)]
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
    def onebody_energy(self,determinant_matrix,SLbasis,permutations_bra_a,permutations_bra_b,permutations_ket_a,permutations_ket_b):
        """Unlike the previous method, the left and right coefficients are now the WHOLE set (occ+unocc). The SLbasis is also the whole (for the corresponding system). The permutations contain all information
        about the construction of the coefficients.
        """

        threshold=1e-12
        Linv_a,da,Rinv_a=LDU_decomp(determinant_matrix[0])#,threshold=threshold) #Alpha LdR decomposition
        Linv_b,db,Rinv_b=LDU_decomp(determinant_matrix[1])#,threshold=threshold) #Beta LdR decomposition

        neh=self.number_electronshalf
        num_singularities_left=len(da[np.abs(da)<threshold])
        num_singularities_right=len(db[np.abs(db)<threshold])
        num_singularities=num_singularities_left+num_singularities_right
        if num_singularities>=2:
            return 0
        energy_1e=self.onebody_energy_alt(determinant_matrix,SLbasis,permutations_bra_a,permutations_bra_b,permutations_ket_a,permutations_ket_b)
        '''
        return energy_1e
        La=Linv_a.T
        Lb=Linv_b.T
        Ra=Rinv_a.T
        Rb=Rinv_b.T
        electrons_basis=np.arange(neh)
        electrons_alpha_bra=np.where(electrons_basis==permutations_bra_a[0],permutations_bra_a[1],electrons_basis)
        electrons_beta_bra=np.where(electrons_basis==permutations_bra_b[0],permutations_bra_b[1],electrons_basis)
        electrons_alpha_ket=np.where(electrons_basis==permutations_ket_a[0],permutations_ket_a[1],electrons_basis)
        electrons_beta_ket=np.where(electrons_basis==permutations_ket_b[0],permutations_ket_b[1],electrons_basis)
        Hamiltonian_SLbasis_alpha=SLbasis[np.ix_(electrons_alpha_bra,electrons_alpha_ket)]
        Hamiltonian_SLbasis_beta=SLbasis[np.ix_(electrons_beta_bra,electrons_beta_ket)]
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
        '''
        return energy_1e
    def twobody_energy(self,determinant_matrix,eribasis,permutations_bra_a,permutations_bra_b,permutations_ket_a,permutations_ket_b):
        def case2():
            eri_MO_bbaa=eribasis[np.ix_(electrons_beta_bra,electrons_beta_ket,electrons_alpha_bra,electrons_alpha_ket)]
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
            return energy_2e
        def case1():
            nonlocal num_singularities_right #accesses the outer function's variable
            nonlocal num_singularities_left
            if num_singularities_left==1: #Left matrix is "almost non-invertible"
                try:
                    nonzero_row,nonzero_col=cofactor_index(determinant_matrix[0])
                except TypeError:
                    print("something went wrong finding the cofactor")
                    num_singularities_left=2
                    return case2()
                determinant_0_new=determinant_matrix[0].copy()
                determinant_0_new[nonzero_row,nonzero_col]=determinant_0_new[nonzero_row,nonzero_col]+2
                M1_1,M2_1,M3_1=second_order_adj_matrix_blockdiag_separated(determinant_0_new,determinant_matrix[1])
                determinant_0_new[nonzero_row,nonzero_col]=determinant_0_new[nonzero_row,nonzero_col]-4
                M1_2,M2_2,M3_2=second_order_adj_matrix_blockdiag_separated(determinant_0_new,determinant_matrix[1])
                M1s=0.5*(M1_1+M1_2)
                M2s=0.5*(M2_1+M2_2)
                M3s=0.5*(M3_1+M3_2)
            elif num_singularities_right==1:
                try:
                    nonzero_row,nonzero_col=cofactor_index(determinant_matrix[1])
                except TypeError:
                    print("something went wrong finding the cofactor")

                    num_singularities_right=2
                    return case2()
                determinant_1_new=determinant_matrix[1].copy()
                determinant_1_new[nonzero_row,nonzero_col]=determinant_1_new[nonzero_row,nonzero_col]+2
                M1_1,M2_1,M3_1=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_1_new)
                determinant_1_new[nonzero_row,nonzero_col]=determinant_1_new[nonzero_row,nonzero_col]-4
                M1_2,M2_2,M3_2=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_1_new)
                M1s=0.5*(M1_1+M1_2)
                M2s=0.5*(M2_1+M2_2)
                M3s=0.5*(M3_1+M3_2)
            M1s=0.5*(M1_1+M1_2)
            M2s=0.5*(M2_1+M2_2)
            M3s=0.5*(M3_1+M3_2)
            G1s,G2s,G3s=get_antisymm_element_separated(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,n)
            energy_2e=np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))+np.trace(dot_nb(M3s,G3s))
            return energy_2e
        def case0():
            G1s,G2s,G3s=get_antisymm_element_separated(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,n)
            try:
                M1s,M2s,M3s=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_matrix[1])
            except np.linalg.LinAlgError:
                nonlocal num_singularities_right #accesses the outer function's variable
                nonlocal num_singularities_left
                if abs(prod_a)<abs(prod_b):
                    num_singularities_left=1
                else:
                    num_singularities_right=1
                return case1()
            energy_2e=np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))+np.trace(dot_nb(M3s,G3s))
            return energy_2e
        #return self.twobody_energy_alt(determinant_matrix,eribasis,permutations_left,permutations_right)
        threshold=1e-7
        Linv_a,da,Rinv_a=LDU_decomp(determinant_matrix[0])#,threshold=threshold) #Alpha LdR decomposition
        Linv_b,db,Rinv_b=LDU_decomp(determinant_matrix[1])#,threshold=threshold) #Beta LdR decomposition
        neh=self.number_electronshalf
        num_singularities_left=len(da[np.abs(da)<threshold])
        prod_a=np.prod(da[np.abs(da)>threshold])
        num_singularities_right=len(db[np.abs(db)<threshold])
        prod_b=np.prod(db[np.abs(db)>threshold])
        num_singularities=num_singularities_left+num_singularities_right
        if num_singularities>=3:
            return 0

        neh=self.number_electronshalf
        n=int(self.number_electronshalf*2)
        nh=self.number_electronshalf
        electrons_basis=np.arange(neh)
        electrons_alpha_bra=np.where(electrons_basis==permutations_bra_a[0],permutations_bra_a[1],electrons_basis)
        electrons_beta_bra=np.where(electrons_basis==permutations_bra_b[0],permutations_bra_b[1],electrons_basis)
        electrons_alpha_ket=np.where(electrons_basis==permutations_ket_a[0],permutations_ket_a[1],electrons_basis)
        electrons_beta_ket=np.where(electrons_basis==permutations_ket_b[0],permutations_ket_b[1],electrons_basis)
        eri_MO_aaaa=eribasis[np.ix_(electrons_alpha_bra,electrons_alpha_ket,electrons_alpha_bra,electrons_alpha_ket)]
        eri_MO_aabb=eribasis[np.ix_(electrons_alpha_bra,electrons_alpha_ket,electrons_beta_bra,electrons_beta_ket)]
        eri_MO_bbbb=eribasis[np.ix_(electrons_beta_bra,electrons_beta_ket,electrons_beta_bra,electrons_beta_ket)]
        energy_2e=0
        if num_singularities==1:
            energy_2e=case1()

        elif num_singularities==2:
            energy_2e=case2()
        else:
            energy_2e=case0()

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
class eigvecsolver_RHF_singlesdoubles(eigvecsolver_RHF):
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
        #1. Create all possible permutations
        for i in range(n_occ):
            for j in range(n_occ,basisset_size):
                permutations.append([[0,i],[0,j]]) #This means: i out, j in!. The zero is there as a "do nothing" operator (now we are operating two permutations...)
        return permutations
    def create_double_fromsame_tosame(self,expansion_coefficients):
        if self.doubles==False:
            return []
        basisset_size=len(expansion_coefficients[0][:,0])
        n_occ=self.number_electronshalf
        n_unocc=basisset_size-n_occ
        permutations=[]
        #1. Create all possible permutations
        for i in range(n_occ):
            for j in range(n_occ,basisset_size):
                permutations.append([[i,i],[j,j]])
        return permutations

    def create_double_fromsame_todiff(self,expansion_coefficients):
        if self.doubles==False:
            return []
        basisset_size=len(expansion_coefficients[0][:,0])
        n_occ=self.number_electronshalf
        n_unocc=basisset_size-n_occ
        permutations=[]
        for i in range(n_occ):
            for l in range(n_occ,basisset_size):
                for k in range(n_occ,l):
                    permutations.append([[i,i],[k,l]])
        return permutations
    def create_double_fromdiff_tosame(self,expansion_coefficients):
        if self.doubles==False:
            return []
        basisset_size=len(expansion_coefficients[0][:,0])
        n_occ=self.number_electronshalf
        n_unocc=basisset_size-n_occ
        permutations=[]
        for i in range(n_occ):
            for j in range(i):
                for l in range(n_occ,basisset_size):
                    permutations.append([[i,j],[l,l]])
        return permutations
    def create_double_fromdiff_todiff(self,expansion_coefficients):
        if self.doubles==False:
            return []
        basisset_size=len(expansion_coefficients[0][:,0])
        n_occ=self.number_electronshalf
        n_unocc=basisset_size-n_occ
        permutations=[]
        for j in range(n_occ):
            for i in range(0,j):
                for l in range(n_occ,basisset_size):
                    for k in range(n_occ,l):
                        permutations.append([[i,j],[k,l]])
        return permutations
    def calculate_energies(self,xc_array,doubles=True):
        self.doubles=doubles #Wether we only want singles, or also doubles. Also serves as a "check" to compare to the other method!!
        """Calculates the molecule's energy"""
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        self.all_new_HF_coefficients=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.sample_points)):
                new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:]) #Actually take the WHOLE thing.
                #new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))[:,:self.number_electronshalf])
            self.all_new_HF_coefficients.append(new_HF_coefficients)
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            try:
                eigval,eigvec=generalized_eigenvector(T,S)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval
            eigval_array.append(eigvec)
            print(eigvec)
        return energy_array,eigval_array
    def calculate_basises(self,mol_xc,n_HF_coef):
        number_electronshalf=self.number_electronshalf
        neh=number_electronshalf
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        e1_slbasis=[]
        e2_slbasis=[]
        for i in range(len(self.HF_coefficients)): #Calculate e1-basis and e2-basis for all combinations of determinants
            temp_1=[]
            temp_2=[]
            for k in range(len(self.HF_coefficients)):
                temp_1.append(np.einsum("ki,lj,kl->ij",n_HF_coef[i],n_HF_coef[k],energy_basis_1e))
                basis=ao2mo.get_mo_eri(energy_basis_2e,(n_HF_coef[i],n_HF_coef[k],n_HF_coef[i],n_HF_coef[k]),aosym="s1")
                temp_2.append(basis)
            e1_slbasis.append(temp_1)
            e2_slbasis.append(temp_2)
        return e1_slbasis, e2_slbasis
    def wv_finder(self,index):
        number_electronshalf=self.number_electronshalf
        neh=number_electronshalf
        state=[]
        if index<self.first: #Just the ground state
            alpha=[[0,0],[0,0]]
            beta=[[0,0],[0,0]]
            state.append([1,alpha,beta])
        elif index<self.single:
            alpha_0=self.all_permutations[index]
            alpha_1=[[0,0],[0,0]]
            beta_1=self.all_permutations[index]
            beta_0=[[0,0],[0,0]]
            state.append([1/np.sqrt(2),alpha_0,beta_0])
            state.append([1/np.sqrt(2),alpha_1,beta_1])
        elif index<self.double_ss:
            alpha=self.all_permutations[index]
            beta=self.all_permutations[index]
            state.append([1,alpha,beta])
        elif index<self.double_sd:
            outs,ins=self.all_permutations[index]
            a,a=outs
            r,s = ins
            alpha_0=[[0,a],[0,r]]
            alpha_1=[[0,a],[0,s]]
            beta_0=[[0,a],[0,s]]
            beta_1=[[0,a],[0,r]]
            state.append([1/np.sqrt(2),alpha_0,beta_0])
            state.append([1/np.sqrt(2),alpha_1,beta_1])
        elif index<self.double_ds:
            outs,ins=self.all_permutations[index]
            a,b=outs
            r,r=ins
            alpha_0=[[0,b],[0,r]]
            alpha_1=[[0,a],[0,r]]
            beta_0=[[0,a],[0,r]]
            beta_1=[[0,a],[0,r]]
            state.append([1/np.sqrt(2),alpha_0,beta_0])
            state.append([1/np.sqrt(2),alpha_1,beta_1])
        elif index<self.double_dd_1:
            outs,ins=self.all_permutations[index]
            a,b=outs
            r,s=ins
            alpha_0=[[a,b],[r,s]]
            beta_0=[[0,0],[0,0]]
            state.append([2/np.sqrt(12),alpha_0,beta_0])
            beta_1=[[a,b],[r,s]]
            alpha_1=[[0,0],[0,0]]
            state.append([2/np.sqrt(12),alpha_1,beta_1])
            alpha_2=[[0,b],[0,r]]
            beta_2=[[0,a],[0,s]]
            state.append([-1/np.sqrt(12),alpha_2,beta_2])
            alpha_3=[[0,b],[0,s]]
            beta_3=[[0,a],[0,r]]
            state.append([1/np.sqrt(12),alpha_3,beta_3])
            alpha_4=[[0,a],[0,r]]
            beta_4=[[0,b],[0,s]]
            state.append([1/np.sqrt(12),alpha_4,beta_4])
            alpha_5=[[0,a],[0,s]]
            beta_5=[[0,b],[0,r]]
            state.append([-1/np.sqrt(12),alpha_5,beta_5])
        elif index<self.double_dd_2:
            outs,ins=self.all_permutations[index]
            a,b=outs
            r,s=ins
            alpha_0=[[0,b],[0,r]]
            beta_0=[[0,a],[0,s]]
            state.append([0.5,alpha_0,beta_0])
            alpha_1=[[0,b],[0,s]]
            beta_1=[[0,a],[0,r]]
            state.append([0.5,alpha_1,beta_1])
            alpha_2=[[0,a],[0,r]]
            beta_2=[[0,b],[0,s]]
            state.append([0.5,alpha_2,beta_2])
            alpha_3=[[0,a],[0,s]]
            beta_3=[[0,b],[0,r]]
            state.append([0.5,alpha_3,beta_3])
        return state
    def calculate_ST_matrices(self,mol_xc,n_HF_coef):
        number_electronshalf=self.number_electronshalf
        neh=number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        self.permutations_s=self.create_singles(n_HF_coef)
        #print(self.permutations_s)

        self.permutations_d_samesame=self.create_double_fromsame_tosame(n_HF_coef)
        self.permutations_d_samediff=self.create_double_fromsame_todiff(n_HF_coef)
        self.permutations_d_diffsame=self.create_double_fromdiff_tosame(n_HF_coef)
        self.permutations_d_diffdiff_1=self.create_double_fromdiff_todiff(n_HF_coef)
        self.permutations_d_diffdiff_2=self.create_double_fromdiff_todiff(n_HF_coef)
        len_all_permutations=1+len(self.permutations_s)+len(self.permutations_d_samesame)+len(self.permutations_d_samediff)+len(self.permutations_d_diffsame)+2*len(self.permutations_d_diffdiff_1)
        self.first=1
        self.single=self.first+len(self.permutations_s)
        self.double_ss=self.single+len(self.permutations_d_samesame)
        self.double_sd=self.double_ss+len(self.permutations_d_samediff)
        self.double_ds=self.double_sd+len(self.permutations_d_diffsame)
        self.double_dd_1=self.double_ds+len(self.permutations_d_diffdiff_1)
        self.double_dd_2=self.double_dd_1+len(self.permutations_d_diffdiff_2)
        self.all_permutations=[[[0,0],[0,0]]]+self.permutations_s+self.permutations_d_samesame+self.permutations_d_samediff+self.permutations_d_diffsame+self.permutations_d_diffdiff_1+self.permutations_d_diffdiff_2
        print("first: %d"%self.first)
        print("single: %d,%d"%(len(self.permutations_s),self.single))
        print("double_ss: %d,%d"%(len(self.permutations_d_samesame),self.double_ss))
        print("double_sd: %d,%d"%(len(self.permutations_d_samediff),self.double_sd))
        print("double_ds: %d,%d"%(len(self.permutations_d_diffsame),self.double_ds))
        print("double_dd1: %d,%d"%(len(self.permutations_d_diffdiff_1),self.double_dd_1))
        print("double_dd2: %d, %d"%(len(self.permutations_d_diffdiff_2),self.double_dd_2))
        print("Total: %d"%len(self.all_permutations))
        #sys.exit(1)
        number_matrix_elements=len(n_HF_coef)*len_all_permutations
        S=np.zeros((number_matrix_elements,number_matrix_elements))
        T=np.zeros((number_matrix_elements,number_matrix_elements))
        e1_slbasis,e2_slbasis=self.calculate_basises(mol_xc,n_HF_coef)
        for c1 in range(number_matrix_elements): #For each matrix element
            i=c1//len_all_permutations # The number of the HF determinant
            j=c1%len_all_permutations #Number of the excitation
            bras=self.wv_finder(j)
            for c2 in range(c1,number_matrix_elements):
                k=c2//len_all_permutations
                l=c2%len_all_permutations
                kets=self.wv_finder(l)
                overlap=0
                energy=0
                for bra in bras:
                    for ket in kets:
                        prefac=bra[0]*ket[0]
                        swaps_bra_alpha=list(bra[1])
                        swaps_bra_beta=list(bra[2])
                        swaps_ket_alpha=list(ket[1])
                        swaps_ket_beta=list(ket[2])
                        overlap_part, energy_part=self.overlap_and_energy(n_HF_coef[i],n_HF_coef[k],overlap_basis,e1_slbasis[i][k],e2_slbasis[i][k],swaps_bra_alpha,swaps_bra_beta,swaps_ket_alpha,swaps_ket_beta)
                        overlap+=prefac*overlap_part
                        energy+=prefac*energy_part
                energy+=overlap*mol_xc.energy_nuc()
                S[c1,c2]=S[c2,c1]=overlap
                T[c1,c2]=T[c2,c1]=energy
            print("%d/%d"%(c1,number_matrix_elements))

        print("Done one")
        print(S)
        print(T)
        return S,T
    def overlap_and_energy(self,coefs_bra,coefs_ket,overlap_basis,onebody,twobody,swaps_bra_alpha,swaps_bra_beta,swaps_ket_alpha,swaps_ket_beta,threshold=1e-16):
        #First step: Calculate the determiants for alpha and beta
        neh=self.number_electronshalf
        alpha_bra_indices=np.arange(neh); alpha_bra_indices[swaps_bra_alpha[0]]=swaps_bra_alpha[1]
        beta_bra_indices=np.arange(neh); beta_bra_indices[swaps_bra_beta[0]]=swaps_bra_beta[1]
        alpha_ket_indices=np.arange(neh);alpha_ket_indices[swaps_ket_alpha[0]]=swaps_ket_alpha[1]
        beta_ket_indices=np.arange(neh);beta_ket_indices[swaps_ket_beta[0]]=swaps_ket_beta[1]
        coefs_bra_alpha=coefs_bra[:,alpha_bra_indices]
        coefs_bra_beta=coefs_bra[:,beta_bra_indices]
        coefs_ket_alpha=coefs_ket[:,alpha_ket_indices]
        coefs_ket_beta=coefs_ket[:,beta_ket_indices]

        detmat_alpha=self.getdeterminant_matrix(overlap_basis,coefs_bra_alpha,coefs_ket_alpha)
        detmat_beta=self.getdeterminant_matrix(overlap_basis,coefs_bra_beta,coefs_ket_beta)
        determinant_matrix=[detmat_alpha,detmat_beta]
        #Next step: Calculate LdR to see if we are done.
        Linv_a,da,Rinv_a=LDU_decomp(detmat_alpha)#,threshold=threshold) #Alpha LdR decomposition
        Linv_b,db,Rinv_b=LDU_decomp(detmat_beta)#,threshold=threshold) #Beta LdR decomposition

        num_singularities_left=len(da[np.abs(da)<threshold])
        num_singularities_right=len(db[np.abs(db)<threshold])
        num_singularities=num_singularities_left+num_singularities_right
        if num_singularities>=3: ###THIS IS WRONG, THIS SHOULD BE 3 !!! 
            #Everything is zero, das ist gut :)
            return (0,0)
        #Calculate the "perturbed basis" for eri
        H2=self.twobody_energy(determinant_matrix,twobody,alpha_bra_indices,beta_bra_indices,alpha_ket_indices,beta_ket_indices) #Need to write this function
        if num_singularities==2:
            overlap=0
            energy=H2
            pass
        H1=self.onebody_energy(determinant_matrix,onebody,alpha_bra_indices,beta_bra_indices,alpha_ket_indices,beta_ket_indices)
        if num_singularities==1:
            overlap=0
            energy=H1+H2
            pass
        if num_singularities==0:
            overlap=np.prod(da)*np.prod(db)
            energy=H1+H2
        return overlap,energy
    def onebody_energy(self,determinant_matrix,SLbasis,alpha_bra_indices,beta_bra_indices,alpha_ket_indices,beta_ket_indices):
        """Unlike the previous method, the left and right coefficients are now the WHOLE set (occ+unocc). The SLbasis is also the whole (for the corresponding system). The permutations contain all information
        about the construction of the coefficients.
        """

        neh=self.number_electronshalf
        Hamiltonian_SLbasis_alpha=SLbasis[np.ix_(alpha_bra_indices,alpha_ket_indices)]
        Hamiltonian_SLbasis_beta=SLbasis[np.ix_(beta_bra_indices,beta_ket_indices)]
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
    def twobody_energy(self,determinant_matrix,eribasis,alpha_bra_indices,beta_bra_indices,alpha_ket_indices,beta_ket_indices):
        def case2():
            eri_MO_bbaa=eribasis[np.ix_(beta_bra_indices,beta_ket_indices,alpha_bra_indices,alpha_ket_indices)]
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
            L=scipy.linalg.block_diag(La,Lb)
            R=scipy.linalg.block_diag(Ra,Rb)
            last_column_R2=second_order_compound_col(R,l=l_index,k=k_index)
            last_row_L2=second_order_compound_row(L,j=j_index,i=i_index)
            nonzero_d=d[np.abs(d)>1e-10]
            d_val=np.prod(nonzero_d)
            adjugate_2S=np.outer(last_column_R2,last_row_L2)*d_val
            G=get_antisymm_element_full(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,eri_MO_bbaa,n)
            energy_2e=np.trace(dot_nb(adjugate_2S,G))
            return energy_2e
        def case1():
            nonlocal num_singularities_right #accesses the outer function's variable
            nonlocal num_singularities_left
            if num_singularities_left==1: #Left matrix is "almost non-invertible"
                try:
                    nonzero_row,nonzero_col=cofactor_index(determinant_matrix[0])
                except TypeError:
                    print("something went wrong finding the cofactor")
                    num_singularities_left=2
                    return case2()
                determinant_0_new=determinant_matrix[0].copy()
                determinant_0_new[nonzero_row,nonzero_col]=determinant_0_new[nonzero_row,nonzero_col]+2
                M1_1,M2_1,M3_1=second_order_adj_matrix_blockdiag_separated(determinant_0_new,determinant_matrix[1])
                determinant_0_new[nonzero_row,nonzero_col]=determinant_0_new[nonzero_row,nonzero_col]-4
                M1_2,M2_2,M3_2=second_order_adj_matrix_blockdiag_separated(determinant_0_new,determinant_matrix[1])
                M1s=0.5*(M1_1+M1_2)
                M2s=0.5*(M2_1+M2_2)
                M3s=0.5*(M3_1+M3_2)
            elif num_singularities_right==1:
                try:
                    nonzero_row,nonzero_col=cofactor_index(determinant_matrix[1])
                except TypeError:
                    print("something went wrong finding the cofactor")

                    num_singularities_right=2
                    return case2()
                determinant_1_new=determinant_matrix[1].copy()
                determinant_1_new[nonzero_row,nonzero_col]=determinant_1_new[nonzero_row,nonzero_col]+2
                M1_1,M2_1,M3_1=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_1_new)
                determinant_1_new[nonzero_row,nonzero_col]=determinant_1_new[nonzero_row,nonzero_col]-4
                M1_2,M2_2,M3_2=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_1_new)
                M1s=0.5*(M1_1+M1_2)
                M2s=0.5*(M2_1+M2_2)
                M3s=0.5*(M3_1+M3_2)
            M1s=0.5*(M1_1+M1_2)
            M2s=0.5*(M2_1+M2_2)
            M3s=0.5*(M3_1+M3_2)
            G1s,G2s,G3s=get_antisymm_element_separated(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,n)
            energy_2e=np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))+np.trace(dot_nb(M3s,G3s))
            return energy_2e
        def case0():
            G1s,G2s,G3s=get_antisymm_element_separated(eri_MO_aaaa,eri_MO_bbbb,eri_MO_aabb,n)
            try:
                M1s,M2s,M3s=second_order_adj_matrix_blockdiag_separated(determinant_matrix[0],determinant_matrix[1])
            except np.linalg.LinAlgError:
                nonlocal num_singularities_right #accesses the outer function's variable
                nonlocal num_singularities_left
                if abs(prod_a)<abs(prod_b):
                    num_singularities_left=1
                else:
                    num_singularities_right=1
                return case1()
            energy_2e=np.trace(dot_nb(M1s,G1s))+np.trace(dot_nb(M2s,G2s))+np.trace(dot_nb(M3s,G3s))
            return energy_2e
        #return self.twobody_energy_alt(determinant_matrix,eribasis,permutations_left,permutations_right)
        threshold=1e-7
        Linv_a,da,Rinv_a=LDU_decomp(determinant_matrix[0])#,threshold=threshold) #Alpha LdR decomposition
        Linv_b,db,Rinv_b=LDU_decomp(determinant_matrix[1])#,threshold=threshold) #Beta LdR decomposition
        neh=self.number_electronshalf
        num_singularities_left=len(da[np.abs(da)<threshold])
        prod_a=np.prod(da[np.abs(da)>threshold])
        num_singularities_right=len(db[np.abs(db)<threshold])
        prod_b=np.prod(db[np.abs(db)>threshold])
        num_singularities=num_singularities_left+num_singularities_right
        if num_singularities>=3:
            return 0

        neh=self.number_electronshalf
        n=int(self.number_electronshalf*2)
        nh=self.number_electronshalf
        eri_MO_aaaa=eribasis[np.ix_(alpha_bra_indices,alpha_ket_indices,alpha_bra_indices,alpha_ket_indices)]
        eri_MO_aabb=eribasis[np.ix_(alpha_bra_indices,alpha_ket_indices,beta_bra_indices,beta_ket_indices)]
        eri_MO_bbbb=eribasis[np.ix_(beta_bra_indices,beta_ket_indices,beta_bra_indices,beta_ket_indices)]
        energy_2e=0
        if num_singularities==1:
            energy_2e=case1()

        elif num_singularities==2:
            energy_2e=case2()
        else:
            energy_2e=case0()

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
