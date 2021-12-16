from pyscf import gto, cc,scf, ao2mo,fci
from rccsd_gs import *
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.linalg import block_diag, eig, orth
from numba import jit
from matrix_operations import *
from helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid
basis = 'dzp'
#basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "N 0 0 0; N 0 0 %f"%x
sample_geom_new=np.linspace(1.5,4.5,31)
sample_geom=[[x] for x in sample_geom_new]
print(sample_geom)
energies=[]
for index,x in enumerate(sample_geom):
    print("%d/%d"%(index,len(sample_geom)))
    mol1=gto.Mole()
    mol1.atom=molecule(*x) #take this as a "basis" assumption.
    mol1.basis=basis
    mol1.unit="bohr"
    mol1.spin=0 #Assume closed shell
    mol1.symmetry=True
    mol1.build()
    mf=mol1.RHF().run(verbose=2) #Solve RHF equations to get overlap
    cisolver = fci.FCI(mol1, mf.mo_coeff)
    e, fcivec = cisolver.kernel()
    energies.append(e)
np.save("N2_1.5to4.5_FULLCI_STO6G.npy",energies)
nig=np.load("N2_1.5to4.5_FULLCI_STO6G.npy")
print(nig)
print(E_FCI)
