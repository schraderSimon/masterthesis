import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf
import pyscf
import sys
np.set_printoptions(linewidth=200)

base_pos=1.3
sample_vals=np.linspace(base_pos-0.3,base_pos+0.3,101)

basis_type="STO-3G"
mol0=gto.Mole()
mol0.atom="""H 0 0 0;F 0 0 %f"""%base_pos #take this as a "basis" assumption.
mol0.basis=basis_type
mol0.unit="Angstrom"
mol0.spin=0 #Assume closed shell
mol0.build()
mf=scf.RHF(mol0)
energy=mf.kernel()
overlap=mol0.intor("int1e_ovlp")
basisset_size=mol0.nao_nr()
expansion_coefficients_mol0 = mf.mo_coeff[:, mf.mo_occ > 0.]

overlap_values=np.zeros_like(sample_vals)
overlap_values_2=np.zeros_like(sample_vals)
for xindex in range(len(sample_vals)):
    x=sample_vals[xindex]
    mol1=gto.Mole()
    mol1.atom="""H 0 0 0; F 0 0 %f"""%x #take this as a "basis" assumption.
    mol1.basis=basis_type
    mol1.unit="Angstrom"
    mol1.spin=0 #Assume closed shell
    mol1.build()
    mf=scf.RHF(mol1)
    energy=mf.kernel()
    overlap=mol1.intor("int1e_ovlp")
    basisset_size=mol1.nao_nr()
    expansion_coefficients_mol1 = mf.mo_coeff[:, mf.mo_occ > 0.]
    number_electronshalf=int(mol1.nelectron/2)
    S_matrix=np.zeros((number_electronshalf,number_electronshalf)) #the matrix to take the determinant of...
    S_matrix_full=np.zeros((number_electronshalf*2,number_electronshalf*2))
    """Create a macromolecule simply to fetch out the integrals"""
    mol_overlap=gto.Mole()
    mol_overlap.atom="""H 0 0 0; F 0 0 %f; H 0 0 0; F 0 0 %f"""%(base_pos,x)
    mol_overlap.unit="Angstrom"
    mol_overlap.basis=basis_type
    mol_overlap.spin=0
    mol_overlap.build()
    overlap=mol_overlap.intor("int1e_ovlp")
    overlap_matrix_of_AO_orbitals=overlap[:basisset_size,basisset_size:]
    S_matrix=np.einsum("ab,ai,bj->ij",overlap_matrix_of_AO_orbitals,expansion_coefficients_mol0,expansion_coefficients_mol1)
    overlap_values[xindex]=np.linalg.det(S_matrix)**2 #The power of 2 comes from the observation that we have alpha and beta spin; the S-matrix for both is block-diagonal, and as we're dealing with RHF, the blocks are identical, and we can

def fitterino(x):
    return np.exp(1/(sample_vals[0]-base_pos)**2*np.log(overlap_values[0])*x**2)
plt.plot(sample_vals-base_pos,overlap_values,label="real")
#plt.plot(sample_vals-base_pos,fitterino(sample_vals-base_pos),label="fit")
plt.legend()
plt.xlabel("Displacement x [Ångstrøm]")
plt.ylabel(r"$<HF(0)|HF(x)>$")
print(sample_vals[0]-base_pos)
print(overlap_values[0])
plt.show()
