import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf
import pyscf
import sys
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200,precision=2,suppress=True)


basis_type="STO-3G"
mol0=gto.Mole()
pos_x0=1.5
pos_x1=1.5
pos_xc=1.5
mol0.atom="""H 0 0 0;F 0 0 %f"""%pos_x0 #take this as a "basis" assumption.
mol0.basis=basis_type
mol0.unit="Angstrom"
mol0.spin=0 #Assume closed shell
mol0.build()
mf=scf.RHF(mol0)
energy=mf.kernel()
overlap=mol0.intor("int1e_ovlp")
basisset_size=mol0.nao_nr()
expansion_coefficients_mol0 = mf.mo_coeff[:, mf.mo_occ > 0.]
expansion_coefficients_mol0_complete = mf.mo_coeff[:, mf.mo_occ > -1e-15]
mol1=gto.Mole()
mol1.atom="""H 0 0 0; F 0 0 %f"""%pos_x1 #take this as a "basis" assumption.
mol1.basis=basis_type
mol1.unit="Angstrom"
mol1.spin=0 #Assume closed shell
mol1.build()
mf=scf.RHF(mol1)
energy=mf.kernel()
overlap=mol1.intor("int1e_ovlp")
basisset_size=mol1.nao_nr()
expansion_coefficients_mol1 = mf.mo_coeff[:, mf.mo_occ > 0.]
expansion_coefficients_mol1_complete = mf.mo_coeff[:, mf.mo_occ > -1e-15]
number_electronshalf=int(mol1.nelectron/2)


mol_energy=gto.Mole()
mol_energy.atom="""H 0 0 0; F 0 0 %f; GHOST_H1 0 0 0;GHOST_F1 0 0 %f; GHOST_H2 0 0 0; GHOST_F2 0 0 %f"""%(pos_xc,pos_x0,pos_x1)
b=basis_type
mol_energy.basis={ "H":b,"F":b,"GHOST_H1": gto.basis.load(b, "H"),"GHOST_F1": gto.basis.load(b, "F"),"GHOST_H2": gto.basis.load(b, "H"),"GHOST_F2": gto.basis.load(b, "F")}
mol_energy.spin=0
mol_energy.build()
kin=mol_energy.intor("int1e_kin")
vnuc=mol_energy.intor("int1e_nuc")
energies=kin+vnuc
energy_matrix=energies[basisset_size:2*basisset_size,2*basisset_size:].copy()
overlap=mol_energy.intor("int1e_ovlp")
overlap_matrix_of_AO_orbitals=overlap[basisset_size:2*basisset_size,2*basisset_size:].copy()
S_matrix_overlap=np.zeros((number_electronshalf,number_electronshalf)) #the matrix to take the determinant of
S_matrix_full_energy=np.zeros((number_electronshalf*2,number_electronshalf*2))
"""We consider h(1) for simplicity, but the idea is rather general"""
#Set up the overlap matrix
for i in range(number_electronshalf):
    for j in range(0,number_electronshalf): #The S_matrix is NOT symmetric!!
        matrix_element=np.einsum("ab,a,b->",overlap_matrix_of_AO_orbitals,expansion_coefficients_mol0[:,i],expansion_coefficients_mol1[:,j])
        S_matrix_overlap[i,j]=matrix_element
        S_matrix_full_energy[2*i,2*j]=matrix_element
        S_matrix_full_energy[2*i+1,2*j+1]=matrix_element


"""Approach 1"""
S_matrix_energy=S_matrix_overlap.copy()
print(S_matrix_overlap)
energy=0
for j in range(number_electronshalf):
    S_matrix_energy=S_matrix_overlap.copy() #Re-initiate Energy matrix
    for i in range(number_electronshalf):
            matrix_element=np.einsum("k,l,kl->",expansion_coefficients_mol0[:,i],expansion_coefficients_mol1[:,j],energy_matrix)
            S_matrix_energy[i,j]=matrix_element
    energy_contribution=np.linalg.det(S_matrix_energy)*np.linalg.det(S_matrix_overlap)
    energy+=energy_contribution
energy*=2 #Beta spin part
print("Energy from approach 1: %f"%energy)
#print("Energy from approach 1: %f"%(np.linalg.det(S_matrix_energy)*np.linalg.det(S_matrix_overlap)))

"""Approach 2"""
#Step 1:Write out the Hamiltonian elements of the new matrix. The new matrix is the hamiltonian element between the occupied AND unoccupied MO-basises
Hamiltonian_SLbasis=np.zeros((number_electronshalf,number_electronshalf))
for i in range(number_electronshalf):
    for j in range(number_electronshalf):
        Hamiltonian_SLbasis[i,j]=matrix_element=np.einsum("k,l,kl->",expansion_coefficients_mol0[:,i],expansion_coefficients_mol1[:,j],energy_matrix)
#Step 2: Calculate overlapperino

toteng=0
for b in range(number_electronshalf):
    S_matrix_energy=S_matrix_overlap.copy()
    S_matrix_energy[:,b]=np.zeros(number_electronshalf)
    for a in range(number_electronshalf):
        S_matrix_energy[a,b]=1
        S_matrix_energy[a-1,b]=0
        toteng+=np.linalg.det(S_matrix_energy)*np.linalg.det(S_matrix_overlap)*Hamiltonian_SLbasis[a,b]
toteng*=2#Beta spin part
print("Energy from approach 2: %f"%toteng)
