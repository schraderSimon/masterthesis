import sys
sys.path.append("../libraries")
from rccsd_gs import *

from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = '6-31G*'
#basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
geom_alphas1=np.linspace(1.2,5,77)
geom_alphas=[[x] for x in geom_alphas1]
sample_geom1=geom_alphas1
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
E_WF=evcsolver.solve_WFCCEVC("energy_data/HF_HS.bin")
print(sample_energies)
