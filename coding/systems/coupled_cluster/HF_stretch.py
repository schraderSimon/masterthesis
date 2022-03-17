from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
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

fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[0][1].set_xlabel("distance (Bohr)")
axes[1][1].set_xlabel("distance (Bohr)")

FCI_geoms=[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,5]
#FCI_energies=[-99.55549,-99.972486,-100.113242,-100.146980,-100.138922,-100.115746,-100.088759,-100.062848,-100.040049,-100.006069,-99.986564,-99.973063]
FCI_energies=[-99.61763594843919, -100.02233609903067, -100.15708053879564, -100.18712346217941, -100.17583274725926, -100.14951120146067, -100.11968327651316, -100.09144082667083, -100.0669261028697, -100.03144824225244, -100.0120880423987, -99.99918263311564]

E_FCI_i = interp1d(FCI_geoms,FCI_energies,kind='cubic')

for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):
        sample_geom1=sample_geometry[i][j]
        sample_geom=[[x] for x in sample_geom1]
        sample_geom1=np.array(sample_geom).flatten()
        geom_alphas1=np.linspace(1.5,5,5)
        geom_alphas=[[x] for x in geom_alphas1]

        t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
        E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,run_cc=True,cc_approx=False,type="procrustes")
        print(E_approx,E_CCSDx)
        energy_simen=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,type="procrustes")
        energy_simen_random=solve_removed_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,occs=1,virts=0.3)


        axes[i][j].plot(geom_alphas1,E_CCSDx,label="CCSD")
        axes[i][j].plot(geom_alphas1,E_approx,label="WF-CCEVC")
        axes[i][j].plot(geom_alphas1,energy_simen,label="AMP-CCEVC")
        axes[i][j].plot(geom_alphas1,energy_simen_random,label=r"AMP-CCEVC (p_v=30\%)")
        axes[i][j].plot(sample_geom,sample_energies,"*",label="Sample points")
        axes[i][j].plot(geom_alphas,E_FCI_i(geom_alphas),label="FCI")
handles, labels = axes[-1][-1].get_legend_handles_labels()
fig.legend(handles, labels,loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.82)
plt.savefig("HF_E_631G_FCI.pdf")
plt.show()
