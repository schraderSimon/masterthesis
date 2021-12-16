from rccsd_gs import *
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo,fci
import sys
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.linalg import block_diag, eig, orth
from numba import jit
from matrix_operations import *
from helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid
basis = 'cc-pVDZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
def molecule(x,y):
    return """Be 0 0 0; H -%f 0 0; H %f 0 0"""%(x,y)
refx=(1,1)
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
x=4*np.random.rand(30,2)+2
sample_geom_new=[]
for i in range(30):
    if (x[i,0]+x[i,1]) <=9 or (x[i,0]+x[i,1])>= 11.5:
        sample_geom_new.append([x[i,0],x[i,1]])
sample_geom=np.concatenate((sample_geom_new,[[5.5,6],[6,5.5],[6,6]]))
print(sample_geom)
span=np.linspace(2,6,9)

geom_alphas=[]
for x in span:
    for y in span:
        geom_alphas.append((x,y))
print(geom_alphas)
x, y = np.meshgrid(span,span)

E_FCI=FCI_energy_curve(geom_alphas,basis,molecule,unit="Bohr")
E_FCI=E_FCI.reshape((len(span),len(span)))
np.save("BeH2_2to6_FULLCI.npy",E_FCI)
nig=np.load("BeH2_2to6_FULLCI.npy")
print(nig)
print(E_FCI)
