import pyscf
from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200,precision=2,suppress=True)


molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x

mol=gto.Mole()
mol.atom="""%s"""%(molecule(3))
mol.charge=0
mol.spin=0
mol.unit="AU"
mol.basis="STO-3G"
mol.build()
number_electronshalf=int(mol.nelectron/2)
mf=mol.UHF().run(verbose=2)
energy=mf.kernel()
print(mf.mo_coeff.shape)
print(mf.mo_occ)
alpha=mf.mo_coeff[0]
beta=mf.mo_coeff[1]
print(alpha.shape)
print(beta.shape)
alpha_occ=mf.mo_coeff[0][:, mf.mo_occ[0] > 0.]
beta_occ=mf.mo_coeff[1][:, mf.mo_occ[0] > 0.]
print(alpha_occ)
print(beta_occ)



mol = gto.Mole()
mol.verbose = 4
mol.atom = [
    ["H", (0., 0.,  2.5)],
    ["H", (0., 0., -2.5)],]
mol.basis = 'cc-pvdz'
mol.build()

mf = scf.UHF(mol)

#
# We can modify the initial guess DM to break spin symmetry.
# For UHF/UKS calculation,  the initial guess DM can be a two-item list
# (alpha,beta).  Assigning alpha-DM and beta-DM to different value can break
# the spin symmetry.
#
# In the following example, the funciton get_init_guess returns the
# superposition of atomic density matrices in which the alpha and beta
# components are degenerated.  The degeneracy are destroyed by zeroing out the
# beta 1s,2s components.
#
dm_alpha, dm_beta = mf.get_init_guess()
dm_beta[:2,:2] = 0
dm = (dm_alpha,dm_beta)
mf.kernel(dm)
