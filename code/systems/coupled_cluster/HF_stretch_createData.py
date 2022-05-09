from rccsd_gs import *
import sys
sys.path.append("../libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = '6-31G*'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[[np.linspace(1.5,5,6),np.linspace(1.5,2,6)],[np.linspace(1.5,2,16),np.linspace(4.5,5,6)]]
FCI_geoms=[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,5]
#FCI_energies=[-99.55549,-99.972486,-100.113242,-100.146980,-100.138922,-100.115746,-100.088759,-100.062848,-100.040049,-100.006069,-99.986564,-99.973063]
FCI_energies=[-99.61763594843919, -100.02233609903067, -100.15708053879564, -100.18712346217941, -100.17583274725926, -100.14951120146067, -100.11968327651316, -100.09144082667083, -100.0669261028697, -100.03144824225244, -100.0120880423987, -99.99918263311564]

E_FCI_i = interp1d(FCI_geoms,FCI_energies,kind='cubic')
import pickle
geom_alphas1=np.linspace(1.2,5,77)
geom_alphas=[[x] for x in geom_alphas1]
energy_dict={}
energy_dict["xval"]=geom_alphas1
energy_dict["FCI"]=E_FCI_i(geom_alphas1)
energy_dict["CCSD"]=CCSD_energy_curve(molecule,geom_alphas,basis)
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
energy_dict["AMP"]=energies_AMP
energy_dict["WF"]=energies_WF
energy_dict["AMPred"]=energies_AMPred
energy_dict["samples"]=sample_geometry
energy_dict["energy_samples"]=energies_sample
file="energy_data/HF_stretch.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(energy_dict,f)
