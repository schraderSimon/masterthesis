from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *

basis = 'aug-cc-pVDZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "H 0 0 %f; Be 0 0 0; H 0 0 -%f"%(x,x)
molecule=lambda x:  "F 0 0 %f; F 0 0 0"%(x)
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[[np.linspace(2.5,3,3),np.linspace(1.5,5,6)],[np.linspace(1.5,2,6),np.linspace(1.5,5,6)]]
sample_indices=[[[0,2,4],[0,7,14,21,28,35]],[[0,1,2,3,4,5],[0,7,14,21,28,35]]]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("distance (Bohr)")
axes[1][1].set_xlabel("distance (Bohr)")
freeze_threshold=np.array([[10**(-4),0],[0,0]])
#Oben links: Normal natural orbitals with delocalized sampling points
#Oben rechts: Frozen natural orbitals with delocalized sampling points
#Unten links: Normal natural orbitals with localized sampling points
#Unten rechts: Standard localized procrustes, delocalized
titles=[["Natural","Frozen Natural"],["Natural","Procrustes"]]
for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):
        if i==1 and j==1:
            break
        sample_geom1=sample_geometry[i][j]
        sample_geom=[[x] for x in sample_geom1]
        sample_geom1=np.array(sample_geom).flatten()
        geom_alphas1=np.linspace(2.5,3,5)
        geom_alphas=[[x] for x in geom_alphas1]
        t1s,t2s,l1s,l2s,sample_energies,reference_natorb_list,reference_overlap_list,reference_noons_list=setUpsamples_naturalOrbitals(geom_alphas,molecule,basis_set,freeze_threshold=freeze_threshold[i,j],desired_samples=sample_indices[i][j])
        print("Done getting ts")
        print(sample_energies)
        mindices=[]
        for noon in reference_noons_list:
            mindices.append(np.where(noon>freeze_threshold[i,j])[0][-1])
        truncation=np.max(mindices)+1
        print(truncation,len(noon))
        natorbs,noons,S=get_natural_orbitals(molecule,geom_alphas,basis_set,reference_natorb_list[0],reference_noons_list[0],reference_overlap_list[0])
        print("Done getting natorbs")
        energy_simen=solve_removed_evc2(geom_alphas,molecule,basis_set,natorbs,t1s,t2s,l1s,l2s,mix_states=False,occs=1,virts=0.3,truncation=truncation)
        energy_simen_false=solve_removed_evc2(geom_alphas,molecule,basis_set,natorbs,t1s,t2s,l1s,l2s,mix_states=False,occs=1,virts=1,truncation=truncation)
        E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,natorbs,t1s,t2s,l1s,l2s,run_cc=True,cc_approx=False,tol=3e-8,truncation=truncation)
        print(E_approx,E_CCSDx)
        axes[i][j].plot(geom_alphas1,E_CCSDx,label="CCSD")
        axes[i][j].plot(geom_alphas1,E_approx,label="WF-CCEVC")
        axes[i][j].plot(geom_alphas1,energy_simen,label="AMP-CCEVC")
        axes[i][j].plot(sample_geom,sample_energies,"*",label="Sample points")
        axes[i][j].set_title(titles[i][j])
        sys.exit(1)
i=1
j=1
sample_geom1=sample_geometry[i][j]
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
geom_alphas1=np.linspace(1.5,5,36)
geom_alphas=[[x] for x in geom_alphas1]

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,run_cc=True,cc_approx=False,type="procrustes")
print(E_approx,E_CCSDx)
energy_simen=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,type="procrustes")


axes[i][j].plot(geom_alphas1,E_CCSDx,label="CCSD")
axes[i][j].plot(geom_alphas1,E_approx,label="WF-CCEVC")
axes[i][j].plot(geom_alphas1,energy_simen,label="AMP-CCEVC")
axes[i][j].plot(sample_geom,sample_energies,"*",label="Sample points")
axes[i][j].set_title(titles[i][j])
handles, labels = axes[-1][-1].get_legend_handles_labels()
fig.legend(handles, labels,loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.82)
plt.savefig("HF_natorbs_.pdf")
print(E_CCSDx)
print(E_approx)
plt.show()
