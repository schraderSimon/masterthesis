import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pyscf import gto, scf, cc, ao2mo
import sys
np.set_printoptions(linewidth=300,precision=2,suppress=True)
def signchange(A,B):
    for i in range(A.shape[1]): #For each column
        if A[:,i].T@B[:,i]<0:
            B[:,i]=-1*B[:,i]

def basischange(C_old,overlap_AOs_newnew):
    S_eig,S_U=np.linalg.eigh(overlap_AOs_newnew)
    S_poweronehalf=S_U@np.diag(S_eig**0.5)@S_U.T
    S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
    C_newbasis=S_poweronehalf@C_old #Basis change
    q,r=np.linalg.qr(C_newbasis) #orthonormalise
    return S_powerminusonehalf@q #change back

def build_molecule(x,molecule,basis):
    """Create a molecule object with parameter x"""

    mol=gto.Mole()
    mol.atom="""%s"""%(molecule(x))
    mol.charge=0
    mol.spin=0
    mol.unit="Bohr"
    mol.basis=basis
    mol.build()
    return mol
basis="6-31G*"
xc_array=np.linspace(1.2,4.5,3)
molecule=lambda x: """F 0 0 0; H 0 0 %f"""%x
molecule_name=r"Hydrogen Fluoride"
mol=build_molecule(1.5,molecule,basis)
print("Energy at r=1.5")
mf = scf.RHF(mol).run()

expansion_coefficients_mol= mf.mo_coeff

newmol=build_molecule(1.5,molecule,basis)
newmol_overlap=newmol.intor("int1e_ovlp")
expansion_coefficients_newmol_unchanged=basischange(expansion_coefficients_mol,newmol_overlap)
print(expansion_coefficients_mol)
print(expansion_coefficients_newmol_unchanged)
signchange(expansion_coefficients_mol,expansion_coefficients_newmol_unchanged)
print(expansion_coefficients_newmol_unchanged)
expansion_coefficients_newmol=expansion_coefficients_newmol_unchanged[:,:5]
print(expansion_coefficients_newmol)
print("Energy at 2.0")
newmol_mf=scf.HF(newmol)
print("Energy at 2.0 with 1.5 coefficients")
initial_guess=2*expansion_coefficients_newmol@expansion_coefficients_newmol.T
newmol_mf.init_guess=initial_guess
newmol_mf.max_cycle=0
newmol_mf.kernel() #Do not do any iterations. This works!!
expansion_coefficients_newmol_unchanged=newmol_mf.mo_coeff





#Solve CCSD for 1.5 and get the amplitudes
print("old")
mycc = cc.CCSD(mf).run()
t1=mycc.t1
t2=mycc.t2
myucc=cc.addons.convert_to_uccsd(mycc) #No need for this bitch
t1_spin=cc.addons.spatial2spinorb(t2).shape
print("new")
mycc_new=cc.CCSD(newmol_mf)
mycc_new.max_cycle=0 #Does not work if set to zero....
mycc_new.kernel(mycc.t1, mycc.t2)
print(mycc_new.t1-mycc.t1)


# CCSD density matrix in MO basis
#
dm1s=mycc.make_rdm1()
dm2s=mycc.make_rdm2()

dm1 = mycc_new.make_rdm1()
dm2 = mycc_new.make_rdm2()

print(dm1s-dm1)
print()
print(np.all(np.abs(dm2s-dm2)<1e-10))
print(np.max(np.abs(dm2s-dm2)))
sys.exit(1)

#
# CCSD energy based on density matrices
#
h1 = np.einsum('pi,pq,qj->ij', expansion_coefficients_newmol_unchanged.conj(), mf.get_hcore(), expansion_coefficients_newmol_unchanged)
nmo = expansion_coefficients_newmol_unchanged.shape[1]
eri = ao2mo.kernel(mol, expansion_coefficients_newmol_unchanged, compact=False).reshape([nmo]*4)
E = np.einsum('pq,qp', h1, dm1)
# Note dm2 is transposed to simplify its contraction to integrals
E+= np.einsum('pqrs,pqrs', eri, dm2) * .5
E+= mol.energy_nuc()
print('E(CCSD) = %s, reference %s' % (E, mycc.e_tot))
