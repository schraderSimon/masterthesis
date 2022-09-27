import sys
sys.path.append("../libraries")
from rccsd_gs import *
from func_lib import *
from numba import jit
from machinelearning import *
from matrix_operations import *
from helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid


basis = 'STO-3G'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
def molecule(x,y):
    if abs(x)>abs(y):

        temp=x
        #x=y
        #y=temp
    return """Be 0 0 0; H -%f 0 0; H %f 0 0"""%(x,y)
refx=(2,2)
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
refx=(4,1)
mol=gto.Mole()
mol.atom=molecule(*refx)
mol.basis=basis
mol.unit="bohr"
mol.build()
print(mol.intor("int1e_ovlp"))
det_41=get_reference_determinant(molecule,refx,basis,charge)

refx=(1,4)
mol=gto.Mole()
mol.atom=molecule(*refx)
mol.basis=basis
mol.unit="bohr"
mol.build()
print(mol.intor("int1e_ovlp"))
sys.exit(1)
det_14=get_reference_determinant(molecule,refx,basis,charge)
det_14_new=localize_procrustes(None,det_14,None,reference_determinant,nelec=6)
det_41_new=localize_procrustes(None,det_41,None,reference_determinant,nelec=6)
print(det_14_new-det_41_new)

"""
We see clearly that Procrustes orbitals for two identical molecules are not identical (and they shouldn't be!). This problem
can be fixed by explicitely fixing the molecule function (e.g. x>y in this particular case?) such that only a given subset of geometries is visited.
This however implements are "hard cutoff"...?  This shouldn't be a problem, reducing computational scaling in the case of BEH2?
#This addes however the geometries x=y in some sort of "boundary" region (lol), so that, in some sense, some geometric information can be lost.
#This problem can in principle be ameliorated by doing clever and necessary switchings in the indices of S and H
"""
