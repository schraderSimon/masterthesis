import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto
from pyscf import scf
import pyscf
import sys


distances=np.linspace(1,5,100)
energies_origdist=np.zeros(100)
energies_optimized=np.zeros(100)
basis="STO-3G"
mol=gto.Mole()
mol.atom="""H 0 0 0; H 0 0 1"""
mol.basis=basis
mol.build()
mf = scf.hf.RHF(mol)
mf.max_cycle=0
mf.kernel()
print(mf.mo_coeff)
mf.max_cycle=50
mf.kernel()
dm_init_guess = mf.make_rdm1()
print(mf.mo_coeff)
print(dm_init_guess)
for i in range(100):
    mol=gto.Mole()
    mol.atom="""H 0 0 0; H 0 0 %f"""%distances[i]
    mol.basis=basis
    mol.build()
    mf = scf.hf.RHF(mol)
    energies_optimized[i]=mf.kernel()
    print(mf.mo_coeff)
for i in range(100):
    mol=gto.Mole()
    mol.atom="""H 0 0 0; H 0 0 %f"""%distances[i]
    mol.basis=basis
    mol.build()
    mf = scf.hf.RHF(mol)
    mf.max_cycle=0
    energies_origdist[i]=mf.kernel(dm_init_guess)
    print(mf.mo_coeff)
plt.plot(distances,energies_optimized,label="optimized")
plt.plot(distances,energies_origdist,label="shit")
plt.legend()
plt.show()
