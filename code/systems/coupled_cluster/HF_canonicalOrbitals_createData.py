from rccsd_gs import *
import sys
sys.path.append("../libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *

basis = 'cc-pVDZ'
charge = 0
molecule=lambda x:  "H 0 0 %f; F 0 0 0"%(x)
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[[np.linspace(1.5,5,6),np.linspace(1.5,5,6)],[np.linspace(1.5,2,6),np.linspace(1.5,2,6)]]
sample_indices=[[[3,10,16,26,30,37],[3,10,16,26,30,37]],[[3,4,5,6,7,8],[3,4,5,6,7,8]]]
geom_alphas1=np.concatenate((np.linspace(1.2,2.5,14),np.linspace(2.7,5.0,24)))

geom_alphas=[[x] for x in geom_alphas1]
energy_dict={}
energy_dict["xval"]=geom_alphas1
energy_dict["basis"]=basis
energies_WF=[[],[]]
energies_AMP=[[],[]]
energies_AMPred=[[],[]]
energies_CCSD=[]
energies_sample=[[],[]]
titles=[["Procrustes","Canonical"],["Procrustes","Canonical"]]
virtsval=0.5

for i in range(2):
    for j in range(2):
        print(i,j)
        if j==0:
            sample_geom1=sample_geometry[i][j]
            sample_geom=[[x] for x in sample_geom1]
            sample_geom1=np.array(sample_geom).flatten()
            t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
            evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
            if j==0:
                E_CCSDx=evcsolver.solve_CCSD()
                energies_CCSD=E_CCSDx
            E_WF=evcsolver.solve_WFCCEVC()
            E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
            E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=virtsval)
            energies_WF[i].append(E_WF)
            energies_AMP[i].append(E_AMP_full)
            energies_AMPred[i].append(E_AMP_red)
            energies_sample[i].append(sample_energies)

        elif j==1:
            sample_geom1=sample_geometry[i][j]
            sample_geom=[[x] for x in sample_geom1]
            sample_geom1=np.array(sample_geom).flatten()
            t1s,t2s,l1s,l2s,sample_energies,reference_natorb_list,reference_overlap_list,reference_noons_list=setUpsamples_canonicalOrbitals(geom_alphas,molecule,basis,desired_samples=sample_indices[i][j])
            print("Done getting ts")
            canonical_orbs,orbital_energies,S=get_canonical_orbitals(molecule,geom_alphas,basis,reference_natorb_list[0],reference_noons_list[0],reference_overlap_list[0])
            evcsolver=EVCSolver(geom_alphas,molecule,basis,canonical_orbs,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
            E_WF=evcsolver.solve_WFCCEVC()
            E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
            energies_AMP[i].append(E_AMP_full)
            E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=virtsval)
            energies_AMPred[i].append(E_AMP_red)
            energies_sample[i].append(sample_energies)
            print(energies_WF)
            print(E_WF)
            energies_WF[i].append(E_WF)
energy_dict["AMP"]=energies_AMP
energy_dict["WF"]=energies_WF
energy_dict["AMPred"]=energies_AMPred
energy_dict["CCSD"]=energies_CCSD

energy_dict["samples"]=sample_geometry
energy_dict["energy_samples"]=energies_sample
energy_dict["titles"]=titles

file="energy_data/HF_Canonical_orb_%.2f.bin"%virtsval
import pickle
with open(file,"wb") as f:
    pickle.dump(energy_dict,f)
