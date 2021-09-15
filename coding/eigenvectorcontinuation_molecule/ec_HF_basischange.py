import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, fci,cc,ao2mo, mp, mcscf
import pyscf
import sys
import itertools
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200,precision=4,suppress=True)

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
            mf=mol.RHF().run(verbose=2)
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
        number_electronshalf=self.number_electronshalf
        large_S=np.zeros((self.number_electronshalf*2,self.number_electronshalf*2))
        large_S[:self.number_electronshalf,:self.number_electronshalf]=determinant_matrix.copy()
        large_S[self.number_electronshalf:,self.number_electronshalf:]=determinant_matrix.copy()
        for k in range(number_electronshalf*2):
            for l in range(k+1,number_electronshalf*2):
                largeS_2e=large_S.copy() #Do I really have to do this 16 times...?
                largeS_2e[:,k]=0
                largeS_2e[:,l]=0
                for a in range(number_electronshalf*2):
                    for b in range(number_electronshalf*2):
                        largeS_2e[a,k]=1
                        largeS_2e[b,l]=1
                        largeS_2e[a-1,k]=0
                        largeS_2e[b-1,l]=0
                        if(k<number_electronshalf and l<number_electronshalf and a < number_electronshalf and b< number_electronshalf):
                            energy_2e+=np.linalg.det(largeS_2e)*MO_eri[a,k,b,l]
                        elif(k>=number_electronshalf and l>=number_electronshalf and a >= number_electronshalf and b>= number_electronshalf):
                            energy_2e+=np.linalg.det(largeS_2e)*MO_eri[a-number_electronshalf,k-number_electronshalf,b-number_electronshalf,l-number_electronshalf]
                        elif(k<number_electronshalf and l>=number_electronshalf and a < number_electronshalf and b>= number_electronshalf):
                            energy_2e+=np.linalg.det(largeS_2e)*MO_eri[a,k,b-number_electronshalf,l-number_electronshalf]
                        elif(k>=number_electronshalf and l<number_electronshalf and a >= number_electronshalf and b< number_electronshalf):
                            energy_2e+=np.linalg.det(largeS_2e)*MO_eri[a-number_electronshalf,k-number_electronshalf,b,l]
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
class eigensolver_RCI(eigvecsolver_RHF):
    """Extension of the RHF case, where we not only include the RHF matrices, but also the excited matrices. Here, I'll only try to do it with ONE."""
    def __init__(self,sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False,number_excitations=1):
        super().__init__(sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False)
        self.number_excitations=number_excitations
        original_occ=np.array(list(range(self.nMO_h)))
        """
        all_HF=[itertools.combinations(list(range(self.nMO_h)),self.ne_h)]
        self.all_permutations=[]
        for possibiliy in all_HF:
            if len(list(set(range(self.number_electronshalf)).intersection(possibility)))<=number_excitations:
                self.all_permutations.append(possibility)
        print(all_HF_maxdiff)
        """
        """
        HF_old_excited=[]
        for i in range(len(self.HF_coefficients)):
            for permutation in self.all_permutations:
                HF_old_excited.append(self.HF_coefficients[i][:,permutation])
        """
        self.all_permutations=[]
        self.all_permutations.append(original_occ)
        for i in range(self.ne_h):
            for j in range(self.ne_h,self.nMO_h):
                newarray=np.copy(original_occ)
                newarray[i], newarray[j]=original_occ[j],original_occ[i]
                self.all_permutations.append(newarray)
    def calculate_energies(self,xc_array):
        """Calculates the molecule's energy"""
        print("enrgie")
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
        eri_MO_bbaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_aaaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_bbbb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
        energy_2e=0
        determinant_matrix_alpha=determinant_matrix[0]
        determinant_matrix_beta=determinant_matrix[1]
        number_electronshalf=self.number_electronshalf
        large_S=np.zeros((number_electronshalf*2,number_electronshalf*2))
        large_S[:number_electronshalf,:number_electronshalf]=determinant_matrix_alpha.copy()
        large_S[number_electronshalf:,number_electronshalf:]=determinant_matrix_beta.copy()
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
                        if(k<number_electronshalf and l<number_electronshalf and a < number_electronshalf and b< number_electronshalf): #alpha, alpha
                            energy_2e+=np.linalg.det(largeS_2e)*eri_MO_aaaa[a,k,b,l]
                        elif(k>=number_electronshalf and l>=number_electronshalf and a >= number_electronshalf and b>= number_electronshalf): #beta, beta
                            energy_2e+=np.linalg.det(largeS_2e)*eri_MO_bbbb[a-number_electronshalf,k-number_electronshalf,b-number_electronshalf,l-number_electronshalf]
                        elif(k<number_electronshalf and l>=number_electronshalf and a < number_electronshalf and b>= number_electronshalf):#alpha, beta
                            energy_2e+=np.linalg.det(largeS_2e)*eri_MO_aabb[a,k,b-number_electronshalf,l-number_electronshalf]
                        elif(k>=number_electronshalf and l<number_electronshalf and a >= number_electronshalf and b< number_electronshalf): #beta,alpha
                            energy_2e+=np.linalg.det(largeS_2e)*eri_MO_bbaa[a-number_electronshalf,k-number_electronshalf,b,l]
        return energy_2e

    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients):
        number_electronshalf=self.number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        S=np.zeros((len(self.HF_coefficients)*2*len(self.all_permutations),len(self.HF_coefficients)*2*len(self.all_permutations)))
        T=np.zeros((len(self.HF_coefficients)*2*len(self.all_permutations),len(self.HF_coefficients)*2*len(self.all_permutations)))
        for i in range(len(self.HF_coefficients)):
            print("i=%d"%i)
            for j in range(i,len(self.HF_coefficients)):
                print("j=%d"%j)
                for k in range(len(self.all_permutations)):
                    print("i=%d,j=%d,k=%d"%(i,j,k))
                    for l in range(len(self.all_permutations)):
                        for a in range(2):
                            for b in range(2):

                                if(a==0):
                                    left_determinant_alpha=new_HF_coefficients[i] #Only beta is perturbed, beta is left in peace
                                    left_determinant_beta=new_HF_coefficients[i][self.all_permutations[k]] #Only alpha is perturbed, beta is left in peace
                                else:
                                    left_determinant_beta=new_HF_coefficients[i] #Only beta is perturbed, beta is left in peace
                                    left_determinant_alpha=new_HF_coefficients[i][self.all_permutations[k]] #Only alpha is perturbed, beta is left in peace
                                if(b==0):
                                    right_determinant_alpha=new_HF_coefficients[j] #Only beta is perturbed, beta is left in peace
                                    right_determinant_beta=new_HF_coefficients[j][self.all_permutations[l]] #Only alpha is perturbed, beta is left in peace
                                else:
                                    right_determinant_beta=new_HF_coefficients[j] #Only beta is perturbed, beta is left in peace
                                    right_determinant_alpha=new_HF_coefficients[j][self.all_permutations[l]] #Only alpha is perturbed, beta is left in peace
                                detlef=[left_determinant_alpha,left_determinant_beta]
                                detright=[right_determinant_alpha,right_determinant_beta]
                                determinant_matrix=self.getdeterminant_matrix(overlap_basis,detlef,detright)
                                overlap=self.getoverlap(determinant_matrix)
                                S[i,j]=S[j,i]=overlap
                                nuc_repulsion_energy=mol_xc.energy_nuc()*overlap
                                energy_1e=self.onebody_energy(energy_basis_1e,detlef,detright,determinant_matrix)
                                energy_2e=self.twobody_energy(energy_basis_2e,detlef,detright,determinant_matrix)
                                energy_total=energy_2e+energy_1e+nuc_repulsion_energy
                                T[i*len(self.HF_coefficients)+2*k+a,j*len(self.HF_coefficients)+2*l+b]=energy_total
                                T[j*len(self.HF_coefficients)+2*l+b,i*len(self.HF_coefficients)+2*k+a]=energy_total
        return S,T
class eigensolver_RHF_knowncoefficients(eigvecsolver_RHF):
    def __init__(self,sample_coefficients,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False):
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
        eri_MO_bbaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_aaaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_bbbb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
        energy_2e=0
        determinant_matrix_alpha=determinant_matrix[0]
        determinant_matrix_beta=determinant_matrix[1]
        number_electronshalf=self.number_electronshalf
        large_S=np.zeros((number_electronshalf*2,number_electronshalf*2))
        large_S[:number_electronshalf,:number_electronshalf]=determinant_matrix_alpha.copy()
        large_S[number_electronshalf:,number_electronshalf:]=determinant_matrix_beta.copy()
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
                        if(k<number_electronshalf and l<number_electronshalf and a < number_electronshalf and b< number_electronshalf): #alpha, alpha
                            energy_2e+=np.linalg.det(largeS_2e)*eri_MO_aaaa[a,k,b,l]
                        elif(k>=number_electronshalf and l>=number_electronshalf and a >= number_electronshalf and b>= number_electronshalf): #beta, beta
                            energy_2e+=np.linalg.det(largeS_2e)*eri_MO_bbbb[a-number_electronshalf,k-number_electronshalf,b-number_electronshalf,l-number_electronshalf]
                        elif(k<number_electronshalf and l>=number_electronshalf and a < number_electronshalf and b>= number_electronshalf):#alpha, beta
                            energy_2e+=np.linalg.det(largeS_2e)*eri_MO_aabb[a,k,b-number_electronshalf,l-number_electronshalf]
                        elif(k>=number_electronshalf and l<number_electronshalf and a >= number_electronshalf and b< number_electronshalf): #beta,alpha
                            energy_2e+=np.linalg.det(largeS_2e)*eri_MO_bbaa[a-number_electronshalf,k-number_electronshalf,b,l]
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
    ax,fig=plt.subplots(figsize=(10,10))

    basis="6-31G*"
    sample_x=np.linspace(1.2,2.5,21)
    xc_array=np.linspace(1.2,5.0,41)
    molecule=lambda x: """F 0 0 0; F 0 0 %f"""%x
    molecule_name="F2"
    '''
    basis="6-31G*"
    molecule_name="BeH2"
    sample_x=np.linspace(0,4,5)
    xc_array=np.linspace(0,4,41)
    molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
    '''
    print("FCI")
    #energiesFCI=FCI_energy_curve(xc_array,basis,molecule=molecule)
    print("CCSDT")
    energiesCC=CC_energy_curve(xc_array,basis,molecule=molecule)

    for i in range(1,len(sample_x)+1,4):
        print("Eigvec (%d)"%(i))
        HF=eigvecsolver_RHF(sample_x[:i],basis,molecule=molecule,symmetry="coov")
        energiesEC,eigenvectors=HF.calculate_energies(xc_array)
        plt.plot(xc_array,energiesEC,label="EC (%d points), %s"%(i,basis))
    #In addition:"outliers":
    HF=eigvecsolver_RHF([2.5,3],basis,molecule=molecule)
    energiesEC,eigenvectors=HF.calculate_energies(xc_array)
    plt.plot(xc_array,energiesEC,label="EC (2.5 and 3)")
    print("UHF")

    energiesHF=energy_curve_RHF(xc_array,basis,molecule=molecule)
    ymin=np.amin([energiesHF,energiesCC])
    ymax=np.amax([energiesHF,energiesCC])
    sample_x=sample_x
    xc_array=xc_array
    plt.plot(xc_array,energiesHF,label="RHF,%s"%basis)
    plt.plot(xc_array,energiesCC,label="CCSD(T),%s"%basis)
    #plt.plot(xc_array,energiesFCI,label="FCI,%s"%basis)
    plt.vlines(sample_x,ymin,ymax,linestyle="--",color=["grey"]*len(sample_x),alpha=0.5,label="sample point")
    plt.xlabel("Molecular distance (Bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.title("%s potential energy curve"%molecule_name)
    plt.legend()
    plt.savefig("EC_RHF_%s.png"%molecule_name)
    plt.show()
    plt.plot(xc_array,energiesEC-energiesFCI,label="EC (max)-FCI")
    plt.plot(xc_array,energiesHF-energiesFCI,label="RHF-FCI")
    plt.legend()
    plt.show()

"""
def basis_change(C_old,overlap_between_AOs_newold,overlap_AOs_newnew):
    def inprod(vec1,vec2):
        return np.einsum("a,b,ab->",vec1,vec2,overlap_AOs_newnew)
    Cnew=np.linalg.inv(overlap_AOs_newnew)@overlap_between_AOs_newold@C_old
    C_newnew=np.zeros_like(Cnew)
    for i in range(Cnew.shape[1]):
        vector=Cnew[:,i]
        for j in range(i):
            vector-=C_newnew[:,j]*inprod(vector,C_newnew[:,j])
        vector=vector/np.sqrt(inprod(vector,vector))
        C_newnew[:,i]=vector
    return C_newnew
"""
