from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVTZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "F 0 0 0; H 0 0 %f"%x
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geom1=np.linspace(1.5,5,8)
sample_geom=[[x] for x in sample_geom1]
import pickle
geom_alphas1=np.linspace(1.5,6,91)
geom_alphas=[[x] for x in geom_alphas1]

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
E_WF=evcsolver.solve_WFCCEVC()
