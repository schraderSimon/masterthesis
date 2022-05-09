import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf
import pyscf
import sys
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200)
import time
from numba import jit

def Mcc(M,c):
    #return np.einsum("ij,i,j->",M,c,c,optimize=True)
    return (M@c).T@c
    """
    summy=0
    for i in range(len(c)):
        for j in range(len(c)):
            summy+= M[i,j]*c[i]*c[j]
    return summy
    """
sample_points=np.linspace(0,2,5)
sample_points=np.append(sample_points,[3,4])
evaluation_points=np.linspace(0,5,11)
S=np.zeros((len(sample_points),len(sample_points)),dtype=float)
T=np.zeros_like(S)

"""Calculate HF energy and expansion coefficients"""
mol=gto.Mole()
basis_type="cc-pVDZ"
mol.atom="""H 0 0 0"""
mol.basis=basis_type
mol.spin=1
mol.build()
mf=scf.RHF(mol)
energy=mf.kernel()
overlap=mol.intor("int1e_ovlp")
basisset_size=mol.nao_nr()
expansion_coefficients = mf.mo_coeff[:, mf.mo_occ > 0.]

"""Calculate overlap matrix"""
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
                summy+=overlap_matrix[a,b]*expansion_coefficients[a,0]*expansion_coefficients[b,0]
        S[index1,index2]=summy

"""Calculate T-matrix"""
start=time.time()
eigenvalues=np.zeros_like(evaluation_points)
for point_index,evaluation_point in enumerate(evaluation_points):
    print("%d/%d"%(point_index,len(evaluation_points)))
    for index1 in range(len(sample_points)):
        for index2 in range(index1,len(sample_points)):
            mol=gto.Mole()
            mol.atom="""H 0 0 %f; GHOST1 0 0 %f; GHOST2 0 0 %f"""%(evaluation_point,sample_points[index1],sample_points[index2])
            mol.unit="Angstrom"
            mol.basis={ "H":basis_type,"GHOST1": gto.basis.load(basis_type, "H"),"GHOST2": gto.basis.load(basis_type, "H")}
            mol.spin=1
            mol.build()
            kin=mol.intor("int1e_kin")
            vnuc=mol.intor("int1e_nuc")
            energies=kin+vnuc
            energy_matrix=energies[2*basisset_size:,basisset_size:2*basisset_size] #2-3 part of the matrix
            summy=Mcc(energy_matrix.copy(),expansion_coefficients[:,0].copy())
            T[index1,index2]=summy
            T[index2,index1]=summy
    lowest_eigenvalue,lowest_eigenvector=generalized_eigenvector(T,S,True)
    eigenvalues[point_index]=lowest_eigenvalue
end=time.time()
print(end-start)
plt.title("Eigenvector continuation with Hydrogen atom in %s basis"%basis_type)
plt.plot(evaluation_points,eigenvalues,label="Eigenvectorcontinuation energy",color="red")
plt.plot(sample_points,[energy]*len(sample_points),"o",color="green",label="Sample points (basis center)")
plt.axhline(energy,label="%s energy"%basis_type)
plt.xlabel("Proton position (Angstrom)")
plt.ylabel("Energy (Hartree)")
plt.legend()
plt.savefig("hydrogen.pdf")
plt.show()



"""First step is to calculate the overlap matrix"""
sample_points=np.linspace(0,1,5)
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
        print(T)
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
