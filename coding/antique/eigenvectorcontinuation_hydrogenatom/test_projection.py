import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf

basis="STO-3G"
mol1=gto.Mole()
mol1.atom="""H 0 0 0; H 0 0 1"""
mol1.basis=basis
mol1.build()
mol2=gto.Mole()
mol2.atom="""H 0 0 0; H 0 0 2"""
mol2.basis=basis
mol2.build()

mf = scf.hf.RHF(mol1)
mf.kernel()
dm_init_guess = mf.make_rdm1()
print(scf.addons.project_dm_nr2nr(mol1, dm_init_guess, mol2))
mf = scf.hf.RHF(mol2)
mf.max_cycle=0
mf.kernel(scf.addons.project_dm_nr2nr(mol1, dm_init_guess, mol2))
