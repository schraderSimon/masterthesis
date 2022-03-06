from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *

basis = 'aug-cc-pVTZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x
refx=[1.75]
sample_geometry=[np.linspace(1.5,2,6)]#,np.linspace(1.5,3,15),np.linspace(1.5,5,15)]

fig,axes=plt.subplots(1,max([2,len(sample_geometry)]),sharey=True)
axes[0].set_ylabel("Energy (Hartree)")
FCI_geoms=[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,5]
#FCI_energies=[-99.55549,-99.972486,-100.113242,-100.146980,-100.138922,-100.115746,-100.088759,-100.062848,-100.040049,-100.006069,-99.986564,-99.973063]
FCI_energies=[-99.61763594843919, -100.02233609903067, -100.15708053879564, -100.18712346217941, -100.17583274725926, -100.14951120146067, -100.11968327651316, -100.09144082667083, -100.0669261028697, -100.03144824225244, -100.0120880423987, -99.99918263311564]

E_FCI_i = interp1d(FCI_geoms,FCI_energies,kind='cubic')


freeze_threshold=1e-4
for i in range(len(sample_geometry)):
    sample_geom1=sample_geometry[i]
    sample_geom=[[x] for x in sample_geom1]
    sample_geom1=np.array(sample_geom).flatten()
    geom_alphas1=np.linspace(1.5,3,51)
    geom_alphas=[[x] for x in geom_alphas1]
    t1s,t2s,l1s,l2s,sample_energies,reference_natorb_list,reference_overlap_list,reference_noons_list=setUpsamples_naturalOrbitals(sample_geom,molecule,basis_set,freeze_threshold=freeze_threshold)
    natorbs,noons=get_natural_orbitals(molecule,geom_alphas,basis_set,reference_natorb_list[0],reference_noons_list[0],reference_overlap_list[0])
    mindices=[]
    for noon in reference_noons_list:
        mindices.append(np.where(noon>freeze_threshold)[0][-1])
    truncation=np.max(mindices)
    print(truncation,len(noon))
    E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,natorbs,t1s,t2s,l1s,l2s,run_cc=True,cc_approx=False,tol=3e-8,truncation=truncation)
    print(E_approx,E_CCSDx)
    energy_simen=solve_evc2(geom_alphas,molecule,basis_set,natorbs,t1s,t2s,l1s,l2s,mix_states=False,type=None,weights=None,truncation=truncation)
    #energy_simen_random=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,random_picks=0.1,type="procrustes")


    axes[i].plot(geom_alphas1,E_CCSDx,label="CCSD")
    axes[i].plot(geom_alphas1,E_approx,label="WF-CCEVC")
    axes[i].plot(geom_alphas1,energy_simen,label="AMP-CCEVC")
    #axes[i].plot(geom_alphas1,energy_simen_random,label="AMP-CCEVC (10%)")
    axes[i].plot(sample_geom,sample_energies,"*",label="Sample points")
    axes[i].set_xlabel("distance (Bohr)")
    axes[i].plot(geom_alphas,E_FCI_i(geom_alphas),label="FCI")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')
plt.tight_layout()
plt.savefig("Afrika.pdf")
print(E_CCSDx)
print(E_approx)
#print(energy_simen)
plt.show()
