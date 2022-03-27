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
molecule=lambda x:  "H 0 0 %f; Be 0 0 0; H 0 0 -%f"%(x,x)
molecule=lambda x:  "H 0 0 %f; F 0 0 0"%(x)
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[[np.linspace(1.5,5,6),np.linspace(1.5,5,6)],[np.linspace(1.5,2,6),np.linspace(1.5,5,6)]]
sample_indices=[[[3,10,17,24,31,38],[3,10,17,24,31,38]],[[3,4,5,6,7,8],[3,10,17,24,31,38]]]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("distance (Bohr)")
axes[1][1].set_xlabel("distance (Bohr)")
freeze_threshold=np.array([[0,10**(-4)],[0,0]])
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
        geom_alphas1=np.linspace(1.5,5,36)
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
        evcsolver=EVCSolver(geom_alphas,molecule,basis_set,natorbs,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False,natorb_truncation=truncation)
        E_WF=evcsolver.solve_WFCCEVC()
        E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
        E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=0.5)
        E_CCSDx=evcsolver.solve_CCSD()
        print(E_approx,E_CCSDx)
        axes[i][j].plot(geom_alphas1,E_CCSDx,label="CCSD")
        axes[i][j].plot(geom_alphas1,E_approx,label="WF-CCEVC")
        axes[i][j].plot(geom_alphas1,E_AMP_full,label=r"AMP-CCEVC (50$\%$)")
        axes[i][j].plot(geom_alphas1,E_AMP_red,label=r"AMP-CCEVC")
        axes[i][j].plot(sample_geom,sample_energies,"*",label="Sample points")
        axes[i][j].set_title(titles[i][j])
        print(energy_simen)
        print(energy_simen_exact)
        print(E_CCSDx)
i=1
j=1
sample_geom1=sample_geometry[i][j]
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
geom_alphas1=np.linspace(1.2,5,39)
geom_alphas=[[x] for x in geom_alphas1]

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,run_cc=True,cc_approx=False,type="procrustes")
print(E_approx,E_CCSDx)
energy_simen=solve_removed_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,occs=1,virts=0.4,truncation=truncation)
energy_simen_exact=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,type="procrustes")


axes[i][j].plot(geom_alphas1,E_CCSDx,label="CCSD")
axes[i][j].plot(geom_alphas1,E_approx,label="WF-CCEVC")
axes[i][j].plot(geom_alphas1,energy_simen,label=r"AMP-CCEVC (30$\%$)")
axes[i][j].plot(geom_alphas1,energy_simen_exact,label=r"AMP-CCEVC")
axes[i][j].plot(sample_geom,sample_energies,"*",label="Sample points")
axes[i][j].set_title(titles[i][j])
handles, labels = axes[-1][-1].get_legend_handles_labels()
fig.legend(handles, labels,loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.82)
plt.savefig("HF_natorbs_XXX.pdf")
print(E_CCSDx)
print(E_approx)
plt.show()
