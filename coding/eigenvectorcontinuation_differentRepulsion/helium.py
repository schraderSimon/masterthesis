import scipy as sp
import numpy as np
import pyscf
from pyscf import gto, scf, ci, cc, ao2mo
from ec_HF_basischange import *
helium=gto.Mole()
helium.atom="He 0 0 0"
helium.basis="6-31G"
helium.unit="AU"
helium.build()

mf = scf.RHF(helium)
#print(helium.intor('int2e',aosym="s1"))
mf.kernel()
eri=helium.intor('int2e',aosym="s1")*1
mf._eri = ao2mo.restore(4,eri,2) #update matrix
helium.incore_anyway=True
print(mf._eri)
mf.kernel()
class eigvecsolver_RHF_coupling(eigvecsolver_RHF):
    def solve_HF(self):
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(x)
            mf = scf.RHF(mol)
            eri=mol.intor('int2e',aosym="s1")*x
            mf._eri = ao2mo.restore(1,eri,mol.nao_nr())
            mol.incore_anyway=True
            mf.kernel()
            expansion_coefficients_mol= mf.mo_coeff[:, mf.mo_occ > 0.]
            HF_coefficients.append(expansion_coefficients_mol)
        self.HF_coefficients=HF_coefficients
    def calculate_energies(self,xc_array):
        energy_array=np.zeros(len(xc_array))
        eigval_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            for i in range(len(self.sample_points)):
                new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp")))
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients,xc)
            try:
                eigval,eigvec=generalized_eigenvector(T,S)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval
            eigval_array.append(eigval)
        return energy_array,eigval_array
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients,param_strengh):
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
                eri = mol_xc.intor('int2e',aosym="s1")*param_strengh #2e in atomic basis
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
def RHF_energy_e2strength(strengths,basis_type,molecule):
    energies=[]
    for i,strength in enumerate(strengths):
        mol1=gto.Mole()
        mol1.atom=molecule(0) #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit='AU'
        mol1.spin=0 #Assume closed shell
        mol1.verbose=2
        mol1.build()
        mf=scf.RHF(mol1)
        eri=mol1.intor('int2e')*strength
        mf._eri = ao2mo.restore(1,eri,mol1.nao_nr())
        #mol1.incore_anyway=True
        energy=mf.kernel()
        energies.append(energy)
    return np.array(energies)

basis="6-31G"
def molecule(x):
    return "H 0 0 0; F 0 0 1.5"
strengths=np.linspace(0,1,21)
energies_HF=RHF_energy_e2strength(strengths,basis,molecule)
sample_strengths=np.linspace(0.1,0.3,5)
for i in range(len(sample_strengths)):
    print("Eigvec (%d)"%(i+1))
    HF=eigvecsolver_RHF_coupling(sample_strengths[:i+1],basis,molecule=molecule)
    energiesEC,eigenvectors=HF.calculate_energies(strengths)
    print(energiesEC)
    plt.plot(strengths,energiesEC,label="EC (%d points), %s"%(i+1,basis))
plt.plot(strengths,energies_HF,label="RHF,%s"%basis)
plt.legend()
plt.show()
