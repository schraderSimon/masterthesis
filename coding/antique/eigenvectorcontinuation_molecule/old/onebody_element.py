import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf
import pyscf
import sys
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200,precision=2,suppress=True)


basis_type="cc-pVDZ"
mol0=gto.Mole()
pos_x0=1.6
pos_x1=1.6
pos_xc=1.6
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
mf.dump_scf_summary()
#eri_4fold_mo = pyscf.ao2mo.incore.full(mol1.intor('int2e',aosym="s1"), expansion_coefficients_mol1_complete)

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


#Set up the overlap matrix
S_matrix_overlap=np.einsum("ab,ai,bj->ij",overlap_matrix_of_AO_orbitals,expansion_coefficients_mol0,expansion_coefficients_mol1)
#Step 0:Write out the Hamiltonian elements of the new matrix. The new matrix is the hamiltonian element between the occupied AND unoccupied MO-basises
Hamiltonian_SLbasis=np.einsum("ki,lj,kl->ij",expansion_coefficients_mol0,expansion_coefficients_mol1,energy_matrix)

#Step 0.5: Calculate the repulsion part (this is easz)
nuc_energy_molecule=gto.Mole()
nuc_energy_molecule.atom="""H 0 0 0; F 0 0 %f"""%pos_xc
nuc_energy_molecule.basis=basis_type
nuc_energy_molecule.build()
nuc_repulsion=nuc_energy_molecule.energy_nuc()*np.linalg.det(S_matrix_overlap)**2
"""Approach 1"""
S_matrix_energy=S_matrix_overlap.copy()
energy=0
for j in range(number_electronshalf):
    S_matrix_energy=S_matrix_overlap.copy() #Re-initiate Energy matrix
    for i in range(number_electronshalf):
            S_matrix_energy[i,j]=Hamiltonian_SLbasis[i,j]
    energy_contribution=np.linalg.det(S_matrix_energy)*np.linalg.det(S_matrix_overlap)
    energy+=energy_contribution
energy*=2 #Beta spin part
print("Energy from approach 1: %f"%energy)
#print("Energy from approach 1: %f"%(np.linalg.det(S_matrix_energy)*np.linalg.det(S_matrix_overlap)))

"""Approach 2"""
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

eri = mol_energy.intor('int2e',aosym="s1")
relevant_eri=eri[basisset_size:2*basisset_size,2*basisset_size:,basisset_size:2*basisset_size,2*basisset_size:]

eri_MO_transformed=np.einsum("ka,lb,mi,nj,kmln->aibj",expansion_coefficients_mol0,expansion_coefficients_mol0,expansion_coefficients_mol1,expansion_coefficients_mol1,relevant_eri)
MO_eri=eri_MO_transformed
two_energy=0
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
                    two_energy+=np.linalg.det(largeS_2e)*MO_eri[a,i,b,j]
                elif(i>=number_electronshalf and j>=number_electronshalf and a >= number_electronshalf and b>= number_electronshalf):
                    two_energy+=np.linalg.det(largeS_2e)*MO_eri[a-number_electronshalf,i-number_electronshalf,b-number_electronshalf,j-number_electronshalf]
                elif(i<number_electronshalf and j>=number_electronshalf and a < number_electronshalf and b>= number_electronshalf):
                    two_energy+=np.linalg.det(largeS_2e)*MO_eri[a,i,b-number_electronshalf,j-number_electronshalf]
                elif(i>=number_electronshalf and j<number_electronshalf and a >= number_electronshalf and b< number_electronshalf):
                    two_energy+=np.linalg.det(largeS_2e)*MO_eri[a-number_electronshalf,i-number_electronshalf,b,j]

energy_total=two_energy+toteng+nuc_repulsion
print("Two-energy: %f"%two_energy)
print("Nuclear repulsion: %f"%nuc_repulsion)
print("The total energy is %.3f"%energy_total)
