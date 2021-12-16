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

basis = '6-31G'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
refx=[1]
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geom1=np.linspace(1,2.6,50)
#sample_geom1=[2.5,3.0,6.0]
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
geom_alphas1=np.linspace(2.2,3.2,41)
geom_alphas=[[x] for x in geom_alphas1]
E_FCI=FCI_energy_curve(np.array(geom_alphas).ravel(),basis,molecule,unit="Bohr")


mix_states=True
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=mix_states,type="procrustes")
#energy_simen=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=mix_states,type="procrustes")
E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=mix_states,run_cc=True,cc_approx=False,type="procrustes")
plt.plot(geom_alphas1,E_CCSDx,label="CCSD left")
plt.plot(geom_alphas1,E_approx,label="CCSD WF left")
#plt.plot(geom_alphas1,energy_simen,label="CCSD AMP left")

refx=[4]
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geom1=np.linspace(3,4,30)
#sample_geom1=[2.5,3.0,6.0]
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()

mix_states=True
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=mix_states,type="procrustes")
#energy_simen=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=mix_states,type="procrustes")

E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=mix_states,run_cc=True,cc_approx=False,type="procrustes")
plt.plot(geom_alphas1,E_CCSDx,label="CCSD right")
plt.plot(geom_alphas1,E_approx,label="CCSD WF right")
#plt.plot(geom_alphas1,energy_simen,label="CCSD AMP right")
plt.plot(geom_alphas1,E_FCI,label="Full CI")
plt.ylim(-16.0,-15.0)
plt.legend()
plt.tight_layout()
plt.savefig("BEH2_insertion_ccPVDZ.pdf")
plt.show()
