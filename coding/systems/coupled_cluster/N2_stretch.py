from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVDZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "N 0 0 0; N 0 0 %f"%x
refx=[2.1]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[[np.linspace(1.7,2.5,6),np.linspace(1.7,3.3,6)],[np.linspace(1.7,3.3,16),np.linspace(1.7,4,16)]]
import pickle
geom_alphas1=np.linspace(1.5,6,91)
geom_alphas=[[x] for x in geom_alphas1]
energy_dict={}
energy_dict["xval"]=geom_alphas1
energies_WF=[[],[]]
energies_AMP=[[],[]]
energies_AMPred=[[],[]]
energies_sample=[[],[]]
for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):
        sample_geom1=sample_geometry[i][j]
        sample_geom=[[x] for x in sample_geom1]
        sample_geom1=np.array(sample_geom).flatten()
        t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
        evcsolver=EVCSolver(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
        E_WF=evcsolver.solve_WFCCEVC()
        E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
        E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=0.5)
        energies_WF[i].append(E_WF)

        energies_AMP[i].append(E_AMP_full)
        energies_AMPred[i].append(E_AMP_red)
        energies_sample[i].append(sample_energies)
energy_dict["CCSD"]=CCSD_energy_curve(molecule,geom_alphas,basis)
energy_dict["AMP"]=energies_AMP
energy_dict["WF"]=energies_WF
energy_dict["AMPred"]=energies_AMPred
energy_dict["samples"]=sample_geometry
energy_dict["energy_samples"]=energies_sample
file="energy_data/N2_stretch.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(energy_dict,f)
