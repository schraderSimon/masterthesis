import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, fci,cc
import pyscf
import sys
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200,precision=2,suppress=True)
def calculate_overlap(x0,x1,basis_type):
    mol0=gto.Mole()
    mol0.atom="""H 0 0 0;F 0 0 %f"""%x0
    mol0.basis=basis_type
    mol0.unit="Angstrom"
    mol0.spin=0 #Assume closed shell
    mol0.build()
    mf=scf.RHF(mol0)
    energy=mf.kernel()
    overlap=mol0.intor("int1e_ovlp")
    basisset_size=mol0.nao_nr()
    expansion_coefficients_mol0 = mf.mo_coeff[:, mf.mo_occ > 0.]
    mol1=gto.Mole()
    mol1.atom="""H 0 0 0; F 0 0 %f"""%x1 #take this as a "basis" assumption.
    mol1.basis=basis_type
    mol1.unit="Angstrom"
    mol1.spin=0 #Assume closed shell
    mol1.build()
    mf=scf.RHF(mol1) #Solve RHF equations to get overlap
    energy=mf.kernel()
    basisset_size=mol1.nao_nr()
    expansion_coefficients_mol1 = mf.mo_coeff[:, mf.mo_occ > 0.]
    number_electronshalf=int(mol1.nelectron/2)
    S_matrix=np.zeros((number_electronshalf,number_electronshalf)) #the matrix to take the determinant of...
    mol_overlap=gto.Mole()
    mol_overlap.atom="""H 0 0 0; F 0 0 %f; H 0 0 0; F 0 0 %f"""%(x0,x1)
    mol_overlap.unit="Angstrom"
    mol_overlap.basis=basis_type
    mol_overlap.spin=0
    mol_overlap.build()
    overlap=mol_overlap.intor("int1e_ovlp")
    overlap_matrix_of_AO_orbitals=overlap[:basisset_size,basisset_size:]
    S_matrix=np.einsum("ab,ai,bj->ij",overlap_matrix_of_AO_orbitals,expansion_coefficients_mol0,expansion_coefficients_mol1)
    return np.linalg.det(S_matrix)**2
def calculate_energy(x0,x1,xc,basis_type):
    """Expansion coefficients for WF at x0"""
    mol0=gto.Mole()
    mol0.atom="""H 0 0 0;F 0 0 %f"""%x0 #take this as a "basis" assumption.
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

    """Expansion coefficients for WF at x1"""
    mol1=gto.Mole()
    mol1.atom="""H 0 0 0; F 0 0 %f"""%x1 #take this as a "basis" assumption.
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

    """Overlap matrix of AO orbitals for macro-molec"""
    mol_energy=gto.Mole()
    mol_energy.atom="""H 0 0 0; F 0 0 %f; GHOST_H1 0 0 0;GHOST_F1 0 0 %f; GHOST_H2 0 0 0; GHOST_F2 0 0 %f"""%(xc,x0,x1)
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

    #Step 0.5: Calculate the repulsion part (this is easy)
    nuc_energy_molecule=gto.Mole()
    nuc_energy_molecule.atom="""H 0 0 0; F 0 0 %f"""%xc
    nuc_energy_molecule.basis=basis_type
    nuc_energy_molecule.build()
    nuc_repulsion_energy=nuc_energy_molecule.energy_nuc()*np.linalg.det(S_matrix_overlap)**2 #Only true for RHF

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
    relevant_eri=eri[basisset_size:2*basisset_size,2*basisset_size:,basisset_size:2*basisset_size,2*basisset_size:] #only ghost molecules of interest

    #convert to MO-basis
    eri_MO_transformed=np.einsum("ka,lb,mi,nj,kmln->aibj",expansion_coefficients_mol0,expansion_coefficients_mol0,expansion_coefficients_mol1,expansion_coefficients_mol1,relevant_eri)
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
xvals=np.linspace(0.90,1.0,33)
energies_eigvec_sto3g=np.zeros(len(xvals))
energies_eigvec_ccpVDZ=np.zeros(len(xvals))

sample_x=np.linspace(0.93,0.97,17)
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
print("Calulate overlap")
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        overlap=calculate_overlap(sample_x[i],sample_x[j],"sto-3G")
        S_eigvec_STO3G[i,j]=overlap
        S_eigvec_STO3G[j,i]=overlap
        overlap=calculate_overlap(sample_x[i],sample_x[j],"6-31G")
        S_eigvec_ccpvdz[i,j]=overlap
        S_eigvec_ccpvdz[j,i]=overlap
print("Calculate energies")
for index,xc in enumerate(xvals):
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
