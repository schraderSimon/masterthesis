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
from scipy.interpolate import interp1d
Evangelista_basis = """
F S
9995.0000000 0.001165998577
1506.0000000 0.008875989168
350.3000000 0.042379948279
104.1000000 0.142928825569
34.8400000 0.355371566302
12.2200000 0.462084436069
4.3690000 0.140847828108
F S
12.2200000 -0.148451921450
1.2080000 1.055269441626
F S
0.3634000 1.000000000000
F P
44.3600000 0.020876002398
10.0800000 0.130107014946
2.9960000 0.396166045508
0.9383000 0.620404071267
F P
0.2733000 1.000000000000

H S
19.238400 0.032827991019
2.8987200 0.231203936751
0.6534720 0.817225776436
H S
0.1630642 1.000000
"""

basis = gto.basis.parse(Evangelista_basis)
basis_set = Evangelista_basis
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[np.linspace(1.5,2,5),np.linspace(1.5,3,5),np.linspace(1.5,5,5)]

fig,axes=plt.subplots(1,3,sharey=True)
axes[0].set_ylabel("Energy (Hartree)")
FCI_geoms=[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,5]
FCI_energies=[-99.55549,-99.972486,-100.113242,-100.146980,-100.138922,-100.115746,-100.088759,-100.062848,-100.040049,-100.006069,-99.986564,-99.973063]
E_FCI_i = interp1d(FCI_geoms,FCI_energies,kind='cubic')
for i in range(3):
    sample_geom1=sample_geometry[i]
    sample_geom=[[x] for x in sample_geom1]
    sample_geom1=np.array(sample_geom).flatten()
    geom_alphas1=np.linspace(1.5,5.0,76)
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
    axes[i].plot(geom_alphas1,E_FCI_i(geom_alphas),label="FCI")
    axes[i].set_xlabel("distance (Bohr)")
    #axes[i].legend()
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
plt.savefig("N2_stretch_procrustes.pdf")
plt.show()
