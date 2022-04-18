from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *

basis = 'cc-pVTZ'
basis = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "H 0 0 %f; Be 0 0 0; H 0 0 -%f"%(x,x)
refx=[2]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[[np.linspace(1.5,6.5,6),np.linspace(1.5,6.5,6)],[np.linspace(2.3,2.8,6),np.linspace(2.3,2.8,6)]]
sample_indices=[[[0,10,20,30,40,50],[0,10,20,30,40,50]],[[8,9,10,11,12,13],[8,9,10,11,12,13]]]
geom_alphas1=np.linspace(1.5,6.5,51)
geom_alphas=[[x] for x in geom_alphas1]
freeze_threshold=np.array([[0,10**(-4)],[0,0]])
energy_dict={}
energy_dict["xval"]=geom_alphas1
energy_dict["basis"]=basis
energies_WF=[[],[]]
energies_AMP=[[],[]]
energies_AMPred=[[],[]]
energies_CCSD=[[],[]]
energies_sample=[[],[]]
titles=[["Natural","Frozen Natural"],["Natural","Procrustes"]]
virtsval=0.2

for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):
        if i==1 and j==1:
            break
        sample_geom1=sample_geometry[i][j]
        sample_geom=[[x] for x in sample_geom1]
        sample_geom1=np.array(sample_geom).flatten()

        t1s,t2s,l1s,l2s,sample_energies,reference_natorb_list,reference_overlap_list,reference_noons_list=setUpsamples_naturalOrbitals(geom_alphas,molecule,basis,freeze_threshold=freeze_threshold[i,j],desired_samples=sample_indices[i][j])
        print("Done getting ts")
        print(sample_energies)
        mindices=[]
        for noon in reference_noons_list:
            mindices.append(np.where(noon>freeze_threshold[i,j])[0][-1])
        truncation=np.max(mindices)+1
        print(truncation,len(noon))
        natorbs,noons,S=get_natural_orbitals(molecule,geom_alphas,basis,reference_natorb_list[0],reference_noons_list[0],reference_overlap_list[0])
        print("Done getting natorbs")
        evcsolver=EVCSolver(geom_alphas,molecule,basis,natorbs,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False,natorb_truncation=truncation)
        E_WF=evcsolver.solve_WFCCEVC()
        E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
        E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=virtsval)
        E_CCSDx=evcsolver.solve_CCSD()
        energies_WF[i].append(E_WF)
        energies_AMP[i].append(E_AMP_full)
        energies_AMPred[i].append(E_AMP_red)
        energies_sample[i].append(sample_energies)
        energies_CCSD[i].append(E_CCSDx)
i=1
j=1
sample_geom1=sample_geometry[i][j]
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
E_WF=evcsolver.solve_WFCCEVC()
E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=virtsval)
energies_WF[i].append(E_WF)
energies_AMP[i].append(E_AMP_full)
energies_AMPred[i].append(E_AMP_red)
energies_sample[i].append(sample_energies)
energies_CCSD[i].append(E_CCSDx)



energy_dict["AMP"]=energies_AMP
energy_dict["WF"]=energies_WF
energy_dict["AMPred"]=energies_AMPred
energy_dict["CCSD"]=energies_CCSD

energy_dict["samples"]=sample_geometry
energy_dict["energy_samples"]=energies_sample
energy_dict["titles"]=titles

file="energy_data/BeH2_Natorb_%.2f.bin"%virtsval
import pickle
with open(file,"wb") as f:
    pickle.dump(energy_dict,f)
