import sys
sys.path.append("../libraries")
from rccsd_gs import *
from func_lib import *
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVTZ'
charge = 0
molecule=lambda x:  "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x); molecule_name="HF"
#molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x; molecule_name="HF"

refx=[2]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
if molecule_name=="HF":
    sample_geometry=[[np.linspace(1,5.0,9)]]
    geom_alphas1=np.linspace(1.5,4.5,31)
    virtvals=[1,0.8,0.4,0.2,0.1]
    occvals=[1,0.8,0.6,0.6,0.6]
elif molecule_name=="BeH2":
    sample_geometry=[[np.linspace(1.5,6.0,10)]]
    geom_alphas1=np.linspace(2,5.5,36)
    virtvals=[1,0.8,0.4,0.2,0.1]
    occvals=[1,1,1,1,1]
import pickle

geom_alphas=[[x] for x in geom_alphas1]
energy_dict={}
energy_dict["xval"]=geom_alphas1
energies_WF=[[],[]]
energies_AMP=[[],[]]
energies_AMPred=[[],[]]
energies_sample=[[],[]]
times=[]
niter=[]
projection_errors=[]
for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):
        sample_geom1=sample_geometry[i][j]
        sample_geom=[[x] for x in sample_geom1]
        sample_geom1=np.array(sample_geom).flatten()

        t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
        evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
        for k,virt in enumerate(virtvals):
            E_AMP_red=evcsolver.solve_AMP_CCSD(occs=occvals[k],virts=virt)
            times.append(evcsolver.times)
            niter.append(evcsolver.num_iterations)
            projection_errors.append(evcsolver.projection_errors)
            print(times[-1])
            print(niter[-1])
            print(projection_errors[-1])
energy_dict["samples"]=sample_geometry
energy_dict["times"]=times
energy_dict["niter"]=niter
energy_dict["virtvals"]=virtvals
energy_dict["projection_errors"]=projection_errors
file="energy_data/%s_time.bin"%molecule_name
import pickle
with open(file,"wb") as f:
    pickle.dump(energy_dict,f)
