import sys
sys.path.append("../libraries")
from rccsd_gs import *

from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVTZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=np.linspace(1.5,2,16)
import pickle
geom_alphas1=np.linspace(1.2,5,39)
geom_alphas=[[x] for x in geom_alphas1]
energy_dict={}
sample_geom1=sample_geometry
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
E_WF=evcsolver.solve_WFCCEVC(filename="H_S_test_%s_HF.bin"%basis)
energy_dict={}
energy_dict["xval"]=geom_alphas1
energy_dict["CCSD"]=CCSD_energy_curve(molecule,geom_alphas,basis)
print(list(energy_dict["CCSD"]))
