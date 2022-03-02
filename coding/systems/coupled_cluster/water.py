from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
sys.path.append("../../eigenvectorcontinuation/")
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.optimize import minimize, root,newton
from full_cc import *


def molecule(alpha,r1,r2):
    H2_x=r2*np.cos(alpha)
    H2_y=r2*np.sin(alpha)
    water="O 0 0 0; H %f 0 0; H %f %f 0"%(r1,H2_x,H2_y)
    return water

mix_states=False
basis="STO-6G"
molecule_name="Water"
geom_alphas=[]
sample_geom=[]
number=3
for angle in np.linspace(105,115,number):
    for r1 in np.linspace(1.81,3,number):
        for r2 in np.linspace(1.81,3,number):
            geom_alphas.append([angle,r1,r2])
number=3
"""
for angle in np.linspace(105,107,number):
    for r1 in np.linspace(1.81,2.5,number):
        for r2 in np.linspace(1.81,2.5,number):
            sample_geom.append([angle,r1,r2])
"""
for i in range(10):
    angle=(110 - 105) * np.random.random_sample() + 105
    r1=(2.5-1.5)*np.random.random_sample() + 1.5
    r2=(2.5-1.5)*np.random.random_sample() + 1.5
    sample_geom.append([angle,r1,r2])
ref_x=[105,1.81,1.81]

mol=make_mol(molecule,ref_x,basis,charge=0)
ENUC=mol.energy_nuc()
mf=scf.RHF(mol)
mf.kernel()
rhf_mo_ref=mf.mo_coeff

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,rhf_mo_ref,mix_states)
#x_alphas=np.linspace(0,4,41)
E_CCSD,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis,rhf_mo_ref,t1s,t2s,l1s,l2s,mix_states=mix_states,run_cc=True,cc_approx=True)
energy_simen=solve_evc2(geom_alphas,molecule,basis,rhf_mo_ref,t1s,t2s,l1s,l2s,mix_states=mix_states)
print("ECCSD")
print(E_CCSD)
print("Diffguess")
print(E_diffguess)
print("RHF")
print(E_RHF)
print("E_ownmethod")
print(E_ownmethod)
print("Simen's idea")
print(energy_simen)
print("alpha  , r1  , r2  ,RHF energy, CCSD energy,   Approach1,   Aprroach 2,  approx")
for i, positions in enumerate(geom_alphas):
    print("%5.1f & %2.1f & %2.1f &%10.4f & %10.4f & %10.4f & %10.4f& %10.4f\\\\ \hline"%(positions[0],positions[1],positions[2],E_RHF[i],E_CCSD[i],E_approx[i],energy_simen[i],E_ownmethod[i]))
