import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, fci,cc,ao2mo, mp, mcscf
import pyscf
import sys
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200,precision=4,suppress=True)
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
def basis_change_noProjection(C_old,overlap_between_AOs_newold,overlap_AOs_newnew):
    def inprod(vec1,vec2):
        return np.einsum("a,b,ab->",vec1,vec2,overlap_AOs_newnew)
    Cnew=C_old
    C_newnew=np.zeros_like(Cnew)
    for i in range(Cnew.shape[1]):
        vector=Cnew[:,i]
        for j in range(i):
            vector-=C_newnew[:,j]*inprod(vector,C_newnew[:,j])
        vector=vector/np.sqrt(inprod(vector,vector))
        C_newnew[:,i]=vector
    return C_newnew
def calculate_overlap(x0,x1,xc,basis_type):
    mol0=gto.Mole()
    mol0.atom="""H 0 0 0;F 0 0 %f"""%x0
    mol0.basis=basis_type
    mol0.unit='AU'
    mol0.spin=0 #Assume closed shell
    mol0.build()
    overlap=mol0.intor("int1e_ovlp")
    basisset_size=mol0.nao_nr()
    mf=scf.RHF(mol0)
    mf.kernel()
    expansion_coefficients_mol0_oldbasis = mf.mo_coeff[:, mf.mo_occ > 0.]
    mol1=gto.Mole()
    mol1.atom="""H 0 0 0; F 0 0 %f"""%x1 #take this as a "basis" assumption.
    mol1.basis=basis_type
    mol1.unit='AU'
    mol1.spin=0 #Assume closed shell
    mol1.build()
    mf=scf.RHF(mol1) #Solve RHF equations to get overlap
    mf.kernel()
    expansion_coefficients_mol1_oldbasis = mf.mo_coeff[:, mf.mo_occ > 0.]

    molc_0=gto.Mole()
    molc_0.atom="""H 0 0 0;F 0 0 %f; H 0 0 0; F 0 0 %f"""%(xc,x0)
    molc_0.basis=basis_type
    molc_0.unit="Anstrom"
    molc_0.spin=0
    molc_0.build()

    molc_1=gto.Mole()
    molc_1.atom="""H 0 0 0;F 0 0 %f; H 0 0 0; F 0 0 %f"""%(xc,x1)
    molc_1.basis=basis_type
    molc_1.unit="Anstrom"
    molc_1.spin=0
    molc_1.build()
    overlap_between_AOs_newnew=molc_0.intor("int1e_ovlp")[:basisset_size,:basisset_size] #Top left corner
    overlap_between_AOs_newold_0=molc_0.intor("int1e_ovlp")[:basisset_size,basisset_size:] #Top right corner
    overlap_between_AOs_newold_1=molc_1.intor("int1e_ovlp")[:basisset_size,basisset_size:] #Top right corner
    expansion_coefficients_mol0_newbasis=basis_change_noProjection(expansion_coefficients_mol0_oldbasis,overlap_between_AOs_newold_0,overlap_between_AOs_newnew)
    expansion_coefficients_mol1_newbasis=basis_change_noProjection(expansion_coefficients_mol1_oldbasis,overlap_between_AOs_newold_1,overlap_between_AOs_newnew)

    number_electronshalf=int(mol1.nelectron/2)
    S_matrix=np.zeros((number_electronshalf,number_electronshalf)) #the matrix to take the determinant of...
    S_matrix=np.einsum("ab,ai,bj->ij",overlap_between_AOs_newnew,expansion_coefficients_mol0_newbasis,expansion_coefficients_mol1_newbasis)
    return np.linalg.det(S_matrix)**2

class eigvecsolver_RHF():
    def __init__(self,sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0):
        self.HF_coefficients=[] #The Hartree Fock coefficients solved at the sample points
        self.molecule=molecule
        self.basis_type=basis_type
        self.spin=spin
        self.unit=unit
        self.charge=charge
        self.sample_points=sample_points
        self.solve_HF()
    def solve_HF(self):
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(x)
            mf=mol.RHF().run(verbose=2)
            expansion_coefficients_mol= mf.mo_coeff[:, mf.mo_occ > 0.]
            HF_coefficients.append(expansion_coefficients_mol)
        self.HF_coefficients=HF_coefficients
    def build_molecule(self,x):
        mol=gto.Mole()
        mol.atom="""%s"""%(self.molecule(x))
        mol.charge=self.charge
        mol.spin=self.spin
        mol.unit=self.unit
        mol.symmetry=False
        mol.basis=self.basis_type
        mol.build()
        self.number_electronshalf=int(mol.nelectron/2)
        return mol
    def calculate_energies(self,xc_array):
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.sample_points)):
                new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp")))
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            try:
                eigval,eigvec=generalized_eigenvector(T,S)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval
            eigval_array.append(eigval)
        return energy_array,eigval_array
    def calculate_overlap_matrix(self,overlap_basis,new_HF_coefficients):
        S=np.zeros((len(self.sample_points),len(self.sample_points)))
        for i in range(len(self.sample_points)):
            for j in range(i,len(self.sample_points)):
                determinant_matrix=np.einsum("ab,ai,bj->ij",overlap_basis,new_HF_coefficients[i],new_HF_coefficients[j])
                overlap=np.linalg.det(determinant_matrix)**2
                S[i,j]=overlap
                S[j,i]=overlap
        return S
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients):
        number_electronshalf=self.number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        S=np.zeros((len(self.sample_points),len(self.sample_points)))
        T=np.zeros((len(self.sample_points),len(self.sample_points)))
        for i in range(len(self.sample_points)):
            for j in range(i,len(self.sample_points)):
                determinant_matrix=np.einsum("ab,ai,bj->ij",overlap_basis,new_HF_coefficients[i],new_HF_coefficients[j])
                overlap=np.linalg.det(determinant_matrix)**2
                S[i,j]=overlap
                S[j,i]=overlap

                nuc_repulsion_energy=mol_xc.energy_nuc()*overlap


                Hamiltonian_SLbasis=np.einsum("ki,lj,kl->ij",new_HF_coefficients[i],new_HF_coefficients[j],energy_basis_1e) #Hamilton operator in Slater determinant basis
                energy_1e=0
                for k in range(number_electronshalf):
                    determinant_matrix_energy=determinant_matrix.copy() #Re-initiate Energy matrix
                    for l in range(number_electronshalf):
                            determinant_matrix_energy[l,k]=Hamiltonian_SLbasis[l,k]
                    energy_contribution=np.linalg.det(determinant_matrix_energy)*np.linalg.det(determinant_matrix)
                    energy_1e+=energy_contribution
                energy_1e*=2 #Beta spin part
                eri = mol_xc.intor('int2e',aosym="s1") #2e in atomic basis
                #eri_MO=np.einsum("ka,lb,mi,nj,kmln->aibj",new_HF_coefficients[i],new_HF_coefficients[i],new_HF_coefficients[j],new_HF_coefficients[j],eri)
                MO_eri=ao2mo.get_mo_eri(eri,(new_HF_coefficients[i],new_HF_coefficients[j],new_HF_coefficients[i],new_HF_coefficients[j]))
                #print(MO_eri-eri_MO)
                #MO_eri=eri_MO #notational convenience
                energy_2e=0
                large_S=np.zeros((number_electronshalf*2,number_electronshalf*2))
                large_S[:number_electronshalf,:number_electronshalf]=determinant_matrix.copy()
                large_S[number_electronshalf:,number_electronshalf:]=determinant_matrix.copy()
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

                energy_total=energy_2e+energy_1e+nuc_repulsion_energy
                T[i,j]=energy_total
                T[j,i]=energy_total
        return S,T
    def basischange(self,C_old,overlap_AOs_newnew):
        """
        def inprod(vec1,vec2):
            return np.einsum("a,b,ab->",vec1,vec2,overlap_AOs_newnew)
        C_new=np.zeros_like(C_old)
        for i in range(C_old.shape[1]):
            vector=C_old[:,i]
            for j in range(i):
                vector-=C_new[:,j]*inprod(vector,C_new[:,j])
            vector=vector/np.sqrt(inprod(vector,vector))
            C_new[:,i]=vector
        return C_new
        """
        S_eig,S_U=np.linalg.eigh(overlap_AOs_newnew)
        S_poweronehalf=S_U@np.diag(S_eig**0.5)@S_U.T
        S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
        C_newbasis=S_poweronehalf@C_old #Basis change
        q,r=np.linalg.qr(C_newbasis) #orthonormalise
        return S_powerminusonehalf@q #change back
class eigvecsolver_UHF(eigvecsolver_RHF):
    def __init__(self,sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0):
        super().__init__(sample_points,basis_type,molecule,spin,unit,charge)
    def solve_HF(self):
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(x)
            mf=scf.UHF(mol)
            dm_alpha, dm_beta = mf.get_init_guess()
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
                alpha=self.basischange(self.HF_coefficients[i][0],mol_xc.intor("int1e_ovlp"))
                beta =self.basischange(self.HF_coefficients[i][1],mol_xc.intor("int1e_ovlp"))
                new_HF_coefficients.append([alpha,beta])
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            eigval,eigvec=generalized_eigenvector(T,S)
            energy_array[index]=eigval
            eigval_array.append(eigval)
        return energy_array,eigval_array
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients):
        number_electronshalf=self.number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        S=np.zeros((len(self.sample_points),len(self.sample_points)))
        T=np.zeros((len(self.sample_points),len(self.sample_points)))
        for i in range(len(self.sample_points)):
            for j in range(i,len(self.sample_points)):
                determinant_matrix_alpha=np.einsum("ab,ai,bj->ij",overlap_basis,new_HF_coefficients[i][0],new_HF_coefficients[j][0])
                determinant_matrix_beta=np.einsum("ab,ai,bj->ij",overlap_basis,new_HF_coefficients[i][1],new_HF_coefficients[j][1])
                overlap=np.linalg.det(determinant_matrix_alpha)*np.linalg.det(determinant_matrix_beta)
                S[i,j]=overlap
                S[j,i]=overlap
                nuc_repulsion_energy=mol_xc.energy_nuc()*overlap


                Hamiltonian_SLbasis_alpha=np.einsum("ki,lj,kl->ij",new_HF_coefficients[i][0],new_HF_coefficients[j][0],energy_basis_1e) #Hamilton operator in Slater determinant basis
                Hamiltonian_SLbasis_beta=np.einsum("ki,lj,kl->ij",new_HF_coefficients[i][1],new_HF_coefficients[j][1],energy_basis_1e) #Hamilton operator in Slater determinant basis
                energy_1e=0
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
                eri = mol_xc.intor('int2e',aosym="s1") #2e in atomic basis

                #eri_MO_aabb=np.einsum("ka,lb,mi,nj,kmln->aibj",new_HF_coefficients[i][0],new_HF_coefficients[i][1],new_HF_coefficients[j][0],new_HF_coefficients[j][1],eri)
                eri_MO_aabb=ao2mo.get_mo_eri(eri,(new_HF_coefficients[i][0],new_HF_coefficients[j][0],new_HF_coefficients[i][1],new_HF_coefficients[j][1]),aosym="s1")
                #eri_MO_bbaa=np.einsum("ka,lb,mi,nj,kmln->aibj",new_HF_coefficients[i][1],new_HF_coefficients[i][0],new_HF_coefficients[j][1],new_HF_coefficients[j][0],eri)
                eri_MO_bbaa=ao2mo.get_mo_eri(eri,(new_HF_coefficients[i][1],new_HF_coefficients[j][1],new_HF_coefficients[i][0],new_HF_coefficients[j][0]),aosym="s1")
                eri_MO_aaaa=ao2mo.get_mo_eri(eri,(new_HF_coefficients[i][0],new_HF_coefficients[j][0],new_HF_coefficients[i][0],new_HF_coefficients[j][0]))
                eri_MO_bbbb=ao2mo.get_mo_eri(eri,(new_HF_coefficients[i][1],new_HF_coefficients[j][1],new_HF_coefficients[i][1],new_HF_coefficients[j][1]))
                #eri_MO_aaaa=np.einsum("ka,lb,mi,nj,kmln->aibj",new_HF_coefficients[i][0],new_HF_coefficients[i][0],new_HF_coefficients[j][0],new_HF_coefficients[j][0],eri)
                #eri_MO_bbbb=np.einsum("ka,lb,mi,nj,kmln->aibj",new_HF_coefficients[i][1],new_HF_coefficients[i][1],new_HF_coefficients[j][1],new_HF_coefficients[j][1],eri)
                #print(eri_MO_aabb-neri_MO_aabb)
                energy_2e=0
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

                energy_total=energy_2e+energy_1e+nuc_repulsion_energy
                T[i,j]=energy_total
                T[j,i]=energy_total
        return S,T
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
    mol1.spin=0 #Assume closed shell
    myhf = mol.RHF().run()
    # Use MP2 natural orbitals to define the active space for the single-point CAS-CI calculation
    mymp = mp.UMP2(myhf).run()

    noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
    ncas, nelecas = (6,8)
    mycas = mcscf.CASCI(myhf, ncas, nelecas)
    energyies.append(mycas.kernel(natorbs))
if __name__=="__main__":
    basis="6-31G"
    sample_x=np.flip(np.array([2,2.5,3,3.5,4,4.5,5]))
    xc_array=np.linspace(1,6.0,21)
    molecule=lambda x: """N 0 0 0; N 0 0 %f"""%x
    sample_x=np.linspace(0,4,17)
    sample_x=np.roll(sample_x,1)
    xc_array=np.linspace(0,4,81)
    molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
    print("FCI")
    energiesFCI=FCI_energy_curve(xc_array,basis,molecule=molecule)
    print("CCSDT")
    energiesCC=CC_energy_curve(xc_array,basis,molecule=molecule)

    for i in range(0,7,1):
        print("Eigvec (%d)"%(i+1))
        HF=eigvecsolver_RHF(sample_x[:i+1],basis,molecule=molecule)
        energiesEC,eigenvectors=HF.calculate_energies(xc_array)
        plt.plot(xc_array,energiesEC,label="EC (%d points), %s"%(i+1,basis))

    print("UHF")
    energiesHF=energy_curve_RHF(xc_array,basis,molecule=molecule)
    ymin=np.amin([energiesHF,energiesCC])
    ymax=np.amax([energiesHF,energiesCC])
    sample_x=sample_x
    xc_array=xc_array
    plt.plot(xc_array,energiesHF,label="RHF,%s"%basis)
    plt.plot(xc_array,energiesCC,label="CCSD(T),%s"%basis)
    plt.plot(xc_array,energiesFCI,label="FCI,%s"%basis)
    plt.vlines(sample_x,ymin,ymax,linestyle="--",color=["grey"]*len(sample_x),alpha=0.5,label="sample point")
    plt.xlabel("Molecular distance (Bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.title("Hydrogen Fluoride potential energy curve")
    plt.legend()
    plt.savefig("EC_UHF_itworks.png")
    plt.show()
    plt.plot(xc_array,energiesEC-energiesCC,label="EC (max)-CCSDT")
    #plt.plot(xc_array,energiesCC-energiesCC,label="CC-CCSD(T)")
    plt.plot(xc_array,energiesHF-energiesCC,label="RHF-CCSDT")
    plt.legend()
    plt.show()
