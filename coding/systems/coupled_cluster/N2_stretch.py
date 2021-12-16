from rccsd_gs import *
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo,fci
import sys
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.linalg import block_diag, eig, orth
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVDZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x:  "N 0 0 0; N 0 0 %f"%x
refx=[2]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[np.linspace(1.7,2.5,10),np.linspace(1.7,3.5,10),np.linspace(1.7,4.5,10)]

fig,axes=plt.subplots(1,3,sharey=True)
axes[0].set_ylabel("Energy (Hartree)")
for i in range(3):
    sample_geom1=sample_geometry[i]
    sample_geom=[[x] for x in sample_geom1]
    sample_geom1=np.array(sample_geom).flatten()
    geom_alphas1=np.linspace(1.5,8.0,66)
    geom_alphas=[[x] for x in geom_alphas1]

    t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")

    energy_simen=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,type="procrustes")
    energy_simen_random=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,random_picks=0.1,type="procrustes")

    E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,run_cc=True,cc_approx=False,type="procrustes")
    axes[i].plot(geom_alphas1,E_CCSDx,label="CCSD")
    axes[i].plot(geom_alphas1,E_approx,label="CCSD WF")
    axes[i].plot(geom_alphas1,energy_simen,label="CCSD AMP")
    axes[i].plot(geom_alphas1,energy_simen,label="CCSD AMP (10%)")
    axes[i].plot(sample_geom,sample_energies,"*",label="Sample points")
    axes[i].set_xlabel("distance (Bohr)")
    #axes[i].legend()
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
plt.savefig("N2_stretch_procrustes.pdf")
plt.show()
