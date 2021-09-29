import pyscf
from pyscf import gto, scf,ao2mo
import numpy as np
import matplotlib.pyplot as plt
from eigenvectorcontinuation import *
np.set_printoptions(linewidth=200,precision=6,suppress=True)
def getdeterminant_matrix(AO_overlap,HF_coefficients_left,HF_coefficients_right):
    determinant_matrix=np.einsum("ab,ai,bj->ij",AO_overlap,HF_coefficients_left,HF_coefficients_right)
    return determinant_matrix
def onebody_energy(energy_basis_1e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
    Hamiltonian_SLbasis_alpha=np.einsum("ki,lj,kl->ij",HF_coefficients_left[0],HF_coefficients_right[0],energy_basis_1e) #Hamilton operator in Slater determinant basis
    Hamiltonian_SLbasis_beta=np.einsum("ki,lj,kl->ij",HF_coefficients_left[1],HF_coefficients_right[1],energy_basis_1e) #Hamilton operator in Slater determinant basis
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
def twobody_energy(energy_basis_2e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
    eri_MO_aabb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
    eri_MO_bbaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
    eri_MO_aaaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
    eri_MO_bbbb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
    energy_2e=0
    determinant_matrix_alpha=determinant_matrix[0]
    determinant_matrix_beta=determinant_matrix[1]
    number_electronshalf=neh
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


molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x
mol=gto.Mole()
mol.atom="""%s"""%(molecule(2.0))
mol.basis="6-31G"
mol.unit="bohr"
mol.build()
neh=int(mol.nelectron/2)
mf=mol.RHF().run()
Fao = mf.get_fock()
Fmo = mf.mo_coeff.T @ Fao @ mf.mo_coeff
print(Fmo)
expansion_coefficients_mol= mf.mo_coeff[:, mf.mo_occ >=-1]
switcherino_out=0
switcherino_in=5
energy_basis_1e=mol.intor("int1e_kin")+mol.intor("int1e_nuc")
energy_basis_2e=mol.intor('int2e',aosym="s1")
overlap_basis=mol.intor("int1e_ovlp")
Hamiltonian_SLbasis_beta=np.einsum("ki,lj,kl->ij",expansion_coefficients_mol,expansion_coefficients_mol,energy_basis_1e)
eri_basis=ao2mo.get_mo_eri(energy_basis_2e,(expansion_coefficients_mol,expansion_coefficients_mol,expansion_coefficients_mol,expansion_coefficients_mol),aosym="s1")

for switcherino_in in range(5,6):
    for switcherino_out in range(0,4):
        print(switcherino_out)
        ground_state=expansion_coefficients_mol[:,:neh]
        excited_state=swap_cols(expansion_coefficients_mol,switcherino_out,switcherino_in)[:,:neh]
        HF_coefficients_left=[ground_state,ground_state]
        #HF_coefficients_right=[excited_state,ground_state]
        HF_coefficients_right=[ground_state,excited_state]
        #detmatrix_alpha=getdeterminant_matrix(overlap_basis,ground_state,excited_state)
        detmatrix_alpha=getdeterminant_matrix(overlap_basis,ground_state,ground_state)
        #detmatrix_beta=getdeterminant_matrix(overlap_basis,ground_state,ground_state)
        detmatrix_beta=getdeterminant_matrix(overlap_basis,ground_state,excited_state)
        determinant_matrix=[detmatrix_alpha,detmatrix_beta]
        energy_2e_anal=0
        energy_1e_anal=Hamiltonian_SLbasis_beta[switcherino_out,switcherino_in]
        energy_1e=onebody_energy(energy_basis_1e,HF_coefficients_left,HF_coefficients_right,determinant_matrix)
        energy_2e=twobody_energy(energy_basis_2e,HF_coefficients_left,HF_coefficients_right,determinant_matrix)

        for i in range(5):
            iss=i
            energy_2e_anal+=(2*eri_basis[switcherino_out,switcherino_in,iss,iss]-eri_basis[switcherino_out,iss,iss,switcherino_in])
        print(energy_1e_anal,energy_2e_anal)
        assert(abs(energy_1e_anal+energy_2e_anal)<1e-6)
        print(energy_1e,energy_2e)
        assert(abs(energy_2e+energy_1e)<1e-6)
