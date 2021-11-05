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

    def getdeterminant_matrix(self,AO_overlap,HF_coefficients_left,HF_coefficients_right):
        determinant_matrix=np.einsum("ab,ai,bj->ij",AO_overlap,HF_coefficients_left,HF_coefficients_right)
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
                conv_sol=self.basischange(self.HF_coefficients[i][:,:self.number_electronshalf],mol_xc.intor("int1e_ovlp"))
                determinant_matrix=self.getdeterminant_matrix(overlap_basis,true_HF_coeffmatrix,conv_sol)
                overlap=self.getoverlap(determinant_matrix)
                overlaps[i]=overlap
            overlap_to_HF[index]=np.dot(overlaps,eigvecs[index])
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
                D_tilde=D[np.abs(D)>1e-15]
                S_tilde=np.prod(D_tilde)**2 #Squared because this is RHF.
                S=np.prod(D)**2
                Smat[i,j]=Smat[j,i]=S
                energy=self.energy_func(C_w,C_x,energy_basis_1e,energy_basis_2e,D)*S_tilde
                #energy_naive=self.energy_func_naive(C_w,C_x,energy_basis_1e,energy_basis_2e,D)*S_tilde
                #print(energy-energy_naive)
                nuc_repulsion_energy=mol_xc.energy_nuc()*S
                Tmat[i,j]=Tmat[j,i]=energy+nuc_repulsion_energy
        return Smat,Tmat
    def energy_func_naive(self,C_w,C_x,onebodyp,twobodyp,D):
        twobody=ao2mo.get_mo_eri(twobodyp,(C_w,C_x,C_w,C_x),aosym="s1") #Works well and seems to me to be the correct one.
        onebody=np.einsum("ki,lj,kl->ij",C_w,C_x,onebodyp)
        alpha=np.arange(5)
        beta=np.arange(5)
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
        number_of_zeros=len(D[np.abs(D)<1e-15])*2
        zero_indices=np.where(np.abs(D)<1e-15)
        if number_of_zeros>2:
            return 0
        else:
            P_s=[]
            for i in range(self.number_electronshalf):
                P_s.append(np.einsum("m,v->mv",C_w[:,i],C_x[:,i]))
            P=P_s
            for i in range(self.number_electronshalf):
                P[i]=P[i]/D[i]
            W=np.sum(P,axis=0)
            if number_of_zeros==2:
                print("Bæmp")
                H1=0
                i=zero_indices[0][0]
                J_contr=np.einsum("ms,ms->",P_s[i],np.einsum("vt,msvt->ms",P_s[i],twobody))
                K_contr=np.einsum("ms,ms->",P_s[i],np.einsum("vt,mtvs->ms",P_s[i],twobody))
                H2=J_contr-K_contr
            elif number_of_zeros==0:
                J_contr=4*np.einsum("sm,ms->",W,np.einsum("tv,msvt->ms",W,twobody))
                K_contr=2*np.einsum("sm,ms->",W,np.einsum("tv,mtvs->ms",W,twobody))
                H2=0.5*(J_contr-K_contr)
                H1=2*np.einsum("mv,mv->",W,onebody)
            return H2+H1
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
    def calculate_energies(self,xc_array,doubles=True):
        self.data_creator()
        """Calculates the molecule's energy"""
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        self.all_new_HF_coefficients=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.sample_points)):
                new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp"))) #Actually take the WHOLE thing.
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
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients):
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
                D_tilde_a=D_a[np.abs(D_a)>1e-12]
                D_tilde_b=D_b[np.abs(D_b)>1e-12]
                S_tilde=np.prod(D_tilde_a)*np.prod(D_tilde_b) #Squared because this is RHF.
                S=np.prod(D_a)*np.prod(D_b)
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
        onebody_alpha=np.einsum("ki,lj,kl->ij",C_w_a,C_x_a,onebodyp)
        onebody_beta=np.einsum("ki,lj,kl->ij",C_w_b,C_x_b,onebodyp)
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
    def energy_func(self,C_w_a,C_w_b,C_x_a,C_x_b,onebody,twobody,D_a,D_b,c1=0,c2=0):
        number_of_zeros_alpha=len(D_a[np.abs(D_a)<1e-12])
        number_of_zeros_beta=len(D_a[np.abs(D_b)<1e-12])
        number_of_zeros=number_of_zeros_alpha+number_of_zeros_beta
        if number_of_zeros>2:
            return 0
        zero_indices_alpha=np.where(np.abs(D_a)<1e-12)
        zero_indices_beta=np.where(np.abs(D_b)<1e-12)
        P_s_a=[]
        for i in range(self.number_electronshalf):
            P_s_a.append(np.einsum("m,v->mv",C_w_a[:,i],C_x_a[:,i]))
        P_s_b=[]
        for i in range(self.number_electronshalf):
            P_s_b.append(np.einsum("m,v->mv",C_w_b[:,i],C_x_b[:,i]))
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
                J_contr=np.einsum("sm,ms->",P_s_a[i],np.einsum("tv,msvt->ms",P_s_a[j],twobody,optimize=True),optimize=True)
                K_contr=np.einsum("sm,ms->",P_s_a[i],np.einsum("tv,mtvs->ms",P_s_a[j],twobody,optimize=True),optimize=True)
                H2=J_contr-K_contr
            elif number_of_zeros_beta==2:
                i=zero_indices_beta[0][0]
                j=zero_indices_beta[0][1]
                J_contr=np.einsum("sm,ms->",P_s_b[i],np.einsum("tv,msvt->ms",P_s_b[j],twobody,optimize=True),optimize=True)
                K_contr=np.einsum("sm,ms->",P_s_b[i],np.einsum("tv,mtvs->ms",P_s_b[j],twobody,optimize=True),optimize=True)
                H2=J_contr-K_contr
            else:
                i=zero_indices_alpha[0][0]
                j=zero_indices_beta[0][0]
                J_contr=np.einsum("sm,ms->",P_s_a[i],np.einsum("tv,msvt->ms",P_s_b[j],twobody,optimize=True),optimize=True)
                H2=J_contr
        elif number_of_zeros_alpha==1:
            i=zero_indices_alpha[0][0]
            H1=np.einsum("mv,mv->",P_s_a[i],onebody)
            J_contr=np.einsum("sm,ms->",P_s_a[i],np.einsum("tv,msvt->ms",W_a+W_b,twobody,optimize=True),optimize=True)#+np.einsum("sm,ms->",P_s_a[i],np.einsum("tv,msvt->ms",W_b,twobody))
            K_contr=np.einsum("sm,ms->",P_s_a[i],np.einsum("tv,mtvs->ms",W_a,twobody,optimize=True),optimize=True)
            H2=J_contr-K_contr
        elif number_of_zeros_beta==1:

            i=zero_indices_beta[0][0]
            H1=np.einsum("mv,mv->",P_s_b[i],onebody)
            J_contr=np.einsum("sm,ms->",P_s_b[i],np.einsum("tv,msvt->ms",W_a+W_b,twobody,optimize=True))#+np.einsum("sm,ms->",P_s_b[i],np.einsum("tv,msvt->ms",W_b,twobody))
            K_contr=np.einsum("sm,ms->",P_s_b[i],np.einsum("tv,mtvs->ms",W_b,twobody,optimize=True),optimize=True)
            H2=J_contr-K_contr

        elif number_of_zeros==0:
            W=W_a+W_b
            J_a_a=np.einsum("sm,ms->",W,np.einsum("tv,msvt->ms",W,twobody,optimize=True),optimize=True)
            J_contr=J_a_a
            K_contr=np.einsum("sm,ms->",W_a,np.einsum("tv,mtvs->ms",W_a,twobody,optimize=True),optimize=True)+np.einsum("sm,ms->",W_b,np.einsum("tv,mtvs->ms",W_b,twobody,optimize=True),optimize=True)
            H2=0.5*(J_contr-K_contr)
            H1=np.einsum("mv,mv->",W,onebody,optimize=True)
        return H2+H1
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
