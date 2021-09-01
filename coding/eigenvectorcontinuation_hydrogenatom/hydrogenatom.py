import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf
import pyscf
import sys
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200)
"""First step is to calculate the overlap matrix"""
sample_points=np.linspace(0,1,10)
evaluation_points=np.linspace(0,10,21)
S=np.zeros((len(sample_points),len(sample_points)),dtype=float)
T=np.zeros_like(S)
def do_sto3g():
    for index1,i in enumerate(sample_points):
        for index2,j in enumerate(sample_points):
            mol=gto.Mole()
            mol.atom="""H 0 0 0; GHOST1 0 0 %f; GHOST2 0 0 %f"""%(i,j)
            mol.basis={"GHOST2": gto.basis.load("sto-3g", "H"),"GHOST1": gto.basis.load("sto-3g", "H"), "H":"sto-3g"}
            mol.spin=1
            mol.build()

            kin=mol.intor("int1e_kin")
            vnuc=mol.intor("int1e_nuc")
            overlap=mol.intor("int1e_ovlp")
            overlap_ghostbasis=overlap[1,2]
            S[index1,index2]=overlap_ghostbasis
            energy=kin[0,0]+vnuc[0,0]


    print("Done setting up overlap matrix")
    eigenvalues=np.zeros_like(evaluation_points)
    for point_index,evaluation_point in enumerate(evaluation_points):
        print("%d/%d"%(point_index,len(evaluation_points)))
        for index1,i in enumerate(sample_points):
            for index2,j in enumerate(sample_points):
                mol=gto.Mole()
                mol.atom="""H 0 0 %f; GHOST1 0 0 %f; GHOST2 0 0 %f"""%(evaluation_point,i,j)
                mol.unit="Angstrom"
                mol.basis={"GHOST2": gto.basis.load("sto-3g", "H"),"GHOST1": gto.basis.load("sto-3g", "H"), "H":"sto-3g"}
                mol.spin=1

                mol.build()
                kin=mol.intor("int1e_kin")
                vnuc=mol.intor("int1e_nuc")
                kinetic_energy_of_interest=kin[1,2]
                nuclear_energy_of_interest=vnuc[1,2]
                T[index1,index2]=nuclear_energy_of_interest+kinetic_energy_of_interest
        lowest_eigenvalue,lowest_eigenvector=generalized_eigenvector(T,S,True)
        eigenvalues[point_index]=lowest_eigenvalue
    plt.title("Eigenvector continuation with Hydrogen atom in STO-3G basis")
    plt.plot(evaluation_points,eigenvalues,label="Eigenvectorcontinuation energy",color="red")
    plt.plot(sample_points,[energy]*len(sample_points),"o",color="green",label="Sample points (basis center)")
    plt.axhline(energy,label="STO 3-G energy")
    plt.xlabel("Proton position (Angstrom)")
    plt.ylabel("Energy (Hartree)")
    plt.legend()
    plt.savefig("hydrogen.pdf")
    plt.show()
#do_sto3g()
#sys.exit(1)



"""Try the same thing, but with a much larger basis. Still only Hydrogen, lol"""
sample_points=np.linspace(0,1,3)
evaluation_points=np.linspace(0,100,101)
S=np.zeros((len(sample_points),len(sample_points)),dtype=float)
T=np.zeros_like(S)
mol=gto.Mole()
basis_type="3-21G"
mol.atom="""H 0 0 0"""
mol.basis=basis_type
mol.spin=1
mol.build()
print(mol.intor("int1e_ovlp"))
mf=scf.RHF(mol)
energy=mf.kernel()
overlap=mol.intor("int1e_ovlp")
basisset_size=overlap.shape[0]
expansion_coefficients = mf.mo_coeff[:, mf.mo_occ > 0.]
for index1,i in enumerate(sample_points):
    for index2,j in enumerate(sample_points):
        mol=gto.Mole()
        mol.atom="""H 0 0 %f; H 0 0 %f"""%(i,j)
        #print(i,j)
        mol.unit="Angstrom"
        mol.spin=0
        mol.basis={"H":basis_type}
        mol.build()
        overlap=mol.intor("int1e_ovlp")
        overlap_matrix=overlap[basisset_size:,:basisset_size]
        summy=0
        for a in range(basisset_size):
            for b in range(basisset_size):
                summy+=overlap_matrix[a,b]*expansion_coefficients[a]*expansion_coefficients[b]
        S[index1,index2]=summy
print(S) #This is now the correct S, :P
eigenvalues=np.zeros_like(evaluation_points)
for point_index,evaluation_point in enumerate(evaluation_points):
    print("%d/%d"%(point_index,len(evaluation_points)))
    for index1,i in enumerate(sample_points):
        for index2,j in enumerate(sample_points):
            mol=gto.Mole()
            mol.atom="""H 0 0 %f; GHOST1 0 0 %f; GHOST2 0 0 %f"""%(evaluation_point,i,j)
            mol.unit="Angstrom"
            mol.basis={"GHOST2": gto.basis.load(basis_type, "H"),"GHOST1": gto.basis.load(basis_type, "H"), "H":basis_type}
            mol.spin=1

            mol.build()
            kin=mol.intor("int1e_kin")
            vnuc=mol.intor("int1e_nuc")
            kinetic_energy_of_interest=kin[1,2] #Now this does not hold true anymore
            nuclear_energy_of_interest=vnuc[1,2] #Now this does definitely not hold true anymore :P
            T[index1,index2]=nuclear_energy_of_interest+kinetic_energy_of_interest
    lowest_eigenvalue,lowest_eigenvector=generalized_eigenvector(T,S,True)
    eigenvalues[point_index]=lowest_eigenvalue
