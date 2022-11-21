"""
Get approximate AMP-CCEVC amplitudes for HF molecule in small basis
"""
import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from machinelearning import *
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVTZ'
#basis="6-31G*"
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x;molecule_name="HF"
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,0)
sample_geom1=np.linspace(1.5,4,7)
import pickle
geom_alphas1=np.linspace(1.4,4.1,81)
geom_alphas=[[x] for x in geom_alphas1]

sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()


t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)

xtol=1e-8 #Convergence tolerance

E_CCSD=evcsolver.solve_CCSD_previousgeometry(xtol=xtol)
niter_prevGeom=evcsolver.num_iter

evcsolver.solve_CCSD_noProcrustes(xtol=xtol)
niter_CCSD=evcsolver.num_iter

E_AMP_red_10=evcsolver.solve_AMP_CCSD(occs=1,virts=0.1,xtol=xtol)
t1s_reduced=evcsolver.t1s_final
t2s_reduced=evcsolver.t2s_final
evcsolver.solve_CCSD_startguess(t1s_reduced,t2s_reduced,xtol=xtol)
niter_AMP_startguess10=evcsolver.num_iter

E_AMP_red_20=evcsolver.solve_AMP_CCSD(occs=1,virts=0.2,xtol=xtol)
t1s_reduced=evcsolver.t1s_final
t2s_reduced=evcsolver.t2s_final
evcsolver.solve_CCSD_startguess(t1s_reduced,t2s_reduced,xtol=xtol)
niter_AMP_startguess20=evcsolver.num_iter
outdata={}
outdata["basis"]=basis
outdata["molecule_name"]=molecule_name
outdata["sample_geometries"]=sample_geom1
outdata["test_geometries"]=geom_alphas1
outdata["sample_energies"]=sample_energies
outdata["MP2"]=niter_CCSD
outdata["EVC_10"]=niter_AMP_startguess10
outdata["EVC_20"]=niter_AMP_startguess20
outdata["prevGeom"]=niter_prevGeom
outdata["energies_CCSD"]=E_CCSD
outdata["energies_AMP_20"]=E_AMP_red_20
outdata["energies_AMP_10"]=E_AMP_red_10

file="energy_data/HF_AMPCCEVC_%s_%d.bin"%(basis,len(sample_geom1))
import pickle
with open(file,"wb") as f:
    pickle.dump(outdata,f)
sys.exit(1)
