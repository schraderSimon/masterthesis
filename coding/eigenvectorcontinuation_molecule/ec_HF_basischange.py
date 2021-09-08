import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, fci,cc
import pyscf
import sys
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200,precision=2,suppress=True)
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
    mol0.unit="Angstrom"
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
    mol1.unit="Angstrom"
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
"""
xc=1.5
xs=np.linspace(1.3,1.7,101)
overlaps=np.zeros(101)
for i in range(len(xs)):
    x=xs[i]
    overlaps[i]=calculate_overlap(x,1.5,xc,"STO-3G")
plt.plot(xs,overlaps)
print(overlaps)
plt.show()
"""
class eigvecsolver_RHF():
    def __init__(self,sample_points,basis_type,molecule="H 0 0 0;F 0 0 %f",spin=0,unit="Angstrom",charge=0):
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
            mf=mol.RHF().run(verbose=0)
            expansion_coefficients_mol= mf.mo_coeff[:, mf.mo_occ > 0.]
            HF_coefficients.append(expansion_coefficients_mol)
        self.HF_coefficients=HF_coefficients
    def build_molecule(self,x):
        mol=gto.Mole()
        mol.atom="""%s"""%(self.molecule%x)
        mol.charge=self.charge
        mol.spin=self.spin
        mol.unit=self.unit
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
            eigval,eigvec=generalized_eigenvector(T,S)
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
                eri_MO=np.einsum("ka,lb,mi,nj,kmln->aibj",new_HF_coefficients[i],new_HF_coefficients[i],new_HF_coefficients[j],new_HF_coefficients[j],eri)
                MO_eri=eri_MO #notational convenience
                energy_2e=0
                large_S=np.zeros((number_electronshalf*2,number_electronshalf*2))
                large_S[:number_electronshalf,:number_electronshalf]=determinant_matrix.copy()
                large_S[number_electronshalf:,number_electronshalf:]=determinant_matrix.copy()
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
def energy_curve(xvals,basis_type,molecule="""H 0 0 0; F 0 0 %f"""):
    energies=[]
    for x in xvals:
        mol1=gto.Mole()
        mol1.atom=molecule%x #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit="Angstrom"
        mol1.spin=0 #Assume closed shell
        mol1.build()
        mf=mol1.RHF().run(verbose=0) #Solve RHF equations to get overlap
        energy=mf.kernel()
        energies.append(energy)
    return np.array(energies)
def CC_energy_curve(xvals,basis_type,molecule="""H 0 0 0; F 0 0 %f"""):
    energies=[]
    for x in xvals:
        mol1=gto.Mole()
        mol1.atom="""H 0 0 0; F 0 0 %f"""%x #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit="Angstrom"
        mol1.spin=0 #Assume closed shell
        mol1.build()
        mf=mol1.RHF().run(verbose=0) #Solve RHF equations to get overlap
        ccsolver=cc.CCSD(mf).run(verbose=0)
        energy=ccsolver.e_tot
        energies.append(energy)
    return np.array(energies)


basis="6-31G"
sample_x=[0.5,1.0,1.5]
xc_array=np.linspace(0.5,3.0,41)
HF=eigvecsolver_RHF(sample_x,basis)
plt.plot(xc_array,energy_curve(xc_array,basis),label="RHF,%s"%basis)
plt.plot(xc_array,CC_energy_curve(xc_array,basis),label="CC,%s"%basis)

energies,eigenvectors=HF.calculate_energies(xc_array)
plt.plot(xc_array,energies,label="EC, %s"%basis)
plt.legend()
plt.show()
'''

def calculate_energy(x0,x1,xc,basis_type):
    mol0=gto.Mole()
    mol0.atom="""H 0 0 0;F 0 0 %f"""%x0
    mol0.basis=basis_type
    mol0.unit="Angstrom"
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
    mol1.unit="Angstrom"
    mol1.spin=0 #Assume closed shell
    mol1.build()
    mf=scf.RHF(mol1) #Solve RHF equations to get overlap
    mf.kernel()
    number_electronshalf=int(mol1.nelectron/2)
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


    """Overlap matrix of AO orbitals for macro-molec"""
    mol_energy=gto.Mole()
    mol_energy.atom="""H 0 0 0; F 0 0 %f"""%xc
    b=basis_type
    mol_energy.basis={ "H":b,"F":b}
    mol_energy.spin=0
    mol_energy.build()
    kin=mol_energy.intor("int1e_kin")
    vnuc=mol_energy.intor("int1e_nuc")
    energies=kin+vnuc
    energy_matrix=energies

    #Set up the overlap matrix
    S_matrix_overlap=np.einsum("ab,ai,bj->ij",overlap_between_AOs_newnew,expansion_coefficients_mol0_newbasis,expansion_coefficients_mol1_newbasis)
    #Step 0:Write out the Hamiltonian elements of the new matrix. The new matrix is the hamiltonian element between the occupied AND unoccupied MO-basises
    Hamiltonian_SLbasis=np.einsum("ki,lj,kl->ij",expansion_coefficients_mol0_newbasis,expansion_coefficients_mol1_newbasis,energy_matrix)

    #Step 0.5: Calculate the repulsion part (this is easy)
    nuc_repulsion_energy=mol_energy.energy_nuc()*np.linalg.det(S_matrix_overlap)**2 #Only true for RHF

    """1e energy (approach 1)"""
    energy_1e=0
    for j in range(number_electronshalf):
        S_matrix_energy=S_matrix_overlap.copy() #Re-initiate Energy matrix
        for i in range(number_electronshalf):
                S_matrix_energy[i,j]=Hamiltonian_SLbasis[i,j]
        energy_contribution=np.linalg.det(S_matrix_energy)*np.linalg.det(S_matrix_overlap)
        energy_1e+=energy_contribution
    energy_1e*=2 #Beta spin part


    """2e energy"""
    eri = mol_energy.intor('int2e',aosym="s1") #2e in atomic basis
    relevant_eri=eri #only ghost molecules of interest

    #convert to MO-basis
    eri_MO_transformed=np.einsum("ka,lb,mi,nj,kmln->aibj",expansion_coefficients_mol0_newbasis,expansion_coefficients_mol0_newbasis,expansion_coefficients_mol1_newbasis,expansion_coefficients_mol1_newbasis,relevant_eri)
    MO_eri=eri_MO_transformed #notational convenience
    energy_2e=0
    large_S=np.zeros((number_electronshalf*2,number_electronshalf*2))
    large_S[:number_electronshalf,:number_electronshalf]=S_matrix_overlap.copy()
    large_S[number_electronshalf:,number_electronshalf:]=S_matrix_overlap.copy()
    for i in range(number_electronshalf*2):
        for j in range(i+1,number_electronshalf*2):
            largeS_2e=large_S.copy()
            largeS_2e[:,i]=0
            largeS_2e[:,j]=0
            for a in range(number_electronshalf*2):
                for b in range(number_electronshalf*2):
                    largeS_2e[a,i]=1
                    largeS_2e[b,j]=1
                    largeS_2e[a-1,i]=0
                    largeS_2e[b-1,j]=0
                    if(i<number_electronshalf and j<number_electronshalf and a < number_electronshalf and b< number_electronshalf):
                        energy_2e+=np.linalg.det(largeS_2e)*MO_eri[a,i,b,j]
                    elif(i>=number_electronshalf and j>=number_electronshalf and a >= number_electronshalf and b>= number_electronshalf):
                        energy_2e+=np.linalg.det(largeS_2e)*MO_eri[a-number_electronshalf,i-number_electronshalf,b-number_electronshalf,j-number_electronshalf]
                    elif(i<number_electronshalf and j>=number_electronshalf and a < number_electronshalf and b>= number_electronshalf):
                        energy_2e+=np.linalg.det(largeS_2e)*MO_eri[a,i,b-number_electronshalf,j-number_electronshalf]
                    elif(i>=number_electronshalf and j<number_electronshalf and a >= number_electronshalf and b< number_electronshalf):
                        energy_2e+=np.linalg.det(largeS_2e)*MO_eri[a-number_electronshalf,i-number_electronshalf,b,j]

    energy_total=energy_2e+energy_1e+nuc_repulsion_energy
    return energy_total


def energy_curve(xvals,basis_type):
    energies=[]
    for x in xvals:
        mol1=gto.Mole()
        mol1.atom="""H 0 0 0; F 0 0 %f"""%x #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit="Angstrom"
        mol1.spin=0 #Assume closed shell
        mol1.build()
        mf=scf.RHF(mol1) #Solve RHF equations to get overlap
        energy=mf.kernel()
        energies.append(energy)
    return np.array(energies)
def CC_energy_curve(xvals,basis_type):
    energies=[]
    for x in xvals:
        mol1=gto.Mole()
        mol1.atom="""H 0 0 0; F 0 0 %f"""%x #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit="Angstrom"
        mol1.spin=0 #Assume closed shell
        mol1.build()
        mf=mol1.RHF().run() #Solve RHF equations to get overlap
        ccsolver=cc.CCSD(mf).run()
        energy=ccsolver.e_tot
        energies.append(energy)
    return np.array(energies)
xvals=np.linspace(0.90,1.5,19) # 37
energies_eigvec_sto3g=np.zeros(len(xvals))
energies_eigvec_ccpVDZ=np.zeros(len(xvals))

sample_x=np.linspace(0.93,0.97,5) # 9
sample_x=[0.5,1.0,1.5]
H_eigvec_STO3G=np.zeros((len(sample_x),len(sample_x)))
H_eigvec_ccpvdz=np.zeros((len(sample_x),len(sample_x)))

S_eigvec_STO3G=np.zeros((len(sample_x),len(sample_x)))
S_eigvec_ccpvdz=np.zeros((len(sample_x),len(sample_x)))
print("Calculate HF")
energy_curve_STO3G=energy_curve(xvals,"STO-3G")
energy_curve_ccpVDZ=energy_curve(xvals,"6-31G")
print("Calculate CC")
energy_curve_STO3G_CC=CC_energy_curve(xvals,"STO-3G")
energy_curve_ccpVDZ_CC=CC_energy_curve(xvals,"6-31G")


for index,xc in enumerate(xvals):
    print("Calculate overlap")
    for i in range(len(sample_x)):
        for j in range(i,len(sample_x)):
            overlap=calculate_overlap(sample_x[i],sample_x[j],xc,"sto-3G")
            S_eigvec_STO3G[i,j]=overlap
            S_eigvec_STO3G[j,i]=overlap
            overlap=calculate_overlap(sample_x[i],sample_x[j],xc,"6-31G")
            S_eigvec_ccpvdz[i,j]=overlap
            S_eigvec_ccpvdz[j,i]=overlap
    print("Energy at %.2f"%xc)
    for i in range(len(sample_x)):
        for j in range(i,len(sample_x)):
            energy_STO3G=calculate_energy(sample_x[i],sample_x[j],xc,"sto-3G")
            energy_ccpvdz=calculate_energy(sample_x[i],sample_x[j],xc,"6-31G")
            H_eigvec_STO3G[i,j]=energy_STO3G
            H_eigvec_STO3G[j,i]=energy_STO3G
            H_eigvec_ccpvdz[i,j]=energy_ccpvdz
            H_eigvec_ccpvdz[j,i]=energy_ccpvdz
    print(H_eigvec_STO3G)
    print(H_eigvec_ccpvdz)
    eigenval_STO3G,trash=generalized_eigenvector(H_eigvec_STO3G,S_eigvec_STO3G)
    eigenval_ccpvdz,trash=generalized_eigenvector(H_eigvec_ccpvdz,S_eigvec_ccpvdz)
    energies_eigvec_sto3g[index]=eigenval_STO3G
    energies_eigvec_ccpVDZ[index]=eigenval_ccpvdz
plt.xlabel("Distance from F to H [Angstrom]")
plt.ylabel("Energy (Hartree)")
for x in sample_x:
    plt.axvline(x,linestyle="--",color="grey",alpha=0.5)
plt.plot(xvals,energy_curve_STO3G,label="HF, STO-3G")
plt.plot(xvals,energy_curve_ccpVDZ,label="HF, 6-31G")
plt.plot(xvals,energies_eigvec_sto3g,label="EC, STO-3G")
plt.plot(xvals,energies_eigvec_ccpVDZ,label="EC, 6-31G")
plt.plot(xvals,energy_curve_STO3G_CC,label="CCSD,STO-3G")
plt.plot(xvals,energy_curve_ccpVDZ_CC,label="CCSD,6-31G")
plt.legend()
plt.savefig("Oof2.pdf")
plt.show()
'''
