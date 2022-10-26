"""
Get approximate AMP-CCEVC amplitudes for HF molecule in small basis
"""
import sys
sys.path.append("../libraries")
from rccsd_gs import *
from machinelearning import *
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVTZ'
#basis="6-31G*"
def get_U_matrix(x,molecule,basis,reference_determinant):
    U_matrices=[]
    for xval in x:
        mol = gto.Mole()
        mol.atom = molecule(xval)
        mol.basis = basis
        mol.unit="bohr"
        mol.build()
        hf=scf.RHF(mol)
        hf.kernel()
        C=hf.mo_coeff
        C_new=localize_procrustes(mol,hf.mo_coeff,hf.mo_occ,reference_determinant)
        S=mol.intor("int1e_ovlp")
        U_rot=np.real(scipy.linalg.fractional_matrix_power(S,0.5))@C_new
        U_matrices.append(U_rot)
    return U_matrices
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x;molecule_name="HF"
#molecule=lambda x:  "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x);molecule_name="BeH2"
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[[np.linspace(1.5,5,6),np.linspace(1.5,2,6)],[np.linspace(1.5,2,16),np.linspace(4.5,5,6)]]
sample_geom1=np.linspace(1.5,4,7)
import pickle
geom_alphas1=np.linspace(1.4,4.1,81)
geom_alphas=[[x] for x in geom_alphas1]
energies_WF=[[],[]]
energies_AMP=[[],[]]
energies_AMPred=[[],[]]

sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
#Instead of using the geometry, use the unitary U to go from S^{-1/2} to C (geometry-independent!)
#1. step - obtain the sample-U (not too hard)
#2. step - obtain the test- U (little extra coding, but absolutely do-able)

sample_U=get_U_matrix(sample_geom1,molecule,basis,reference_determinant)
target_U=get_U_matrix(geom_alphas1,molecule,basis,reference_determinant)
sample_C=np.array(get_coulomb_matrix(sample_geom1,molecule))
target_C=np.array(get_coulomb_matrix(geom_alphas1,molecule))
sample_C_inv=[]
target_C_inv=[]
for C in sample_C:
    Cinv=1/C;

    Cinv[Cinv==np.inf]=0;
    Cinv[Cinv==-np.inf]=0
    sample_C_inv.append(Cinv)
sample_C_inv=np.array(sample_C_inv)
for C in target_C:
    Cinv=1/C;
    Cinv[Cinv==np.inf]=0;
    Cinv[Cinv==-np.inf]=0
    target_C_inv.append(Cinv)
target_C_inv=np.array(target_C_inv)
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)

"""
Set up machine learning for t amplitudes
"""

t1s_orth,t2s_orth,coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)
t1_machinelearn=[]
t2_machinelearn=[]
t1_machinelearn_U=[]
t2_machinelearn_U=[]
means_distance=[]
means_invdistance=[]

for i in range(len(sample_geom1)):
    mean,std=multivariate_gaussian_gpy(sample_C,coefs[i],target_C,sigma=1,l=1)
    means_distance.append(mean)
    mean,std=multivariate_gaussian_gpy_matrixInput(sample_U,coefs[i],target_U,sigma=1,l=1)
    means_invdistance.append(mean)
means=np.array(means_distance)
means_U=np.array(means_invdistance)

for i in range(len(geom_alphas1)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(coefs)):
        t1_temp+=means[j][i]*t1s_orth[j]
        t2_temp+=means[j][i]*t2s_orth[j]
    t1_machinelearn.append(t1_temp)
    t2_machinelearn.append(t2_temp)
for i in range(len(geom_alphas1)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(coefs)):
        t1_temp+=means_U[j][i]*t1s_orth[j]
        t2_temp+=means_U[j][i]*t2s_orth[j]
    t1_machinelearn_U.append(t1_temp)
    t2_machinelearn_U.append(t2_temp)

print("Initial")
xtol=1e-8 #Convergence tolerance
E_ML=evcsolver.calculate_CCSD_energies_from_guess(t1_machinelearn,t2_machinelearn,xtol=xtol)
E_ML_U=evcsolver.calculate_CCSD_energies_from_guess(t1_machinelearn_U,t2_machinelearn_U,xtol=xtol)

evcsolver.solve_CCSD_startguess(t1_machinelearn,t2_machinelearn,xtol=xtol)
niter_machinelearn_dist=evcsolver.num_iter
evcsolver.solve_CCSD_startguess(t1_machinelearn_U,t2_machinelearn_U,xtol=xtol)
niter_machinelearn_U=evcsolver.num_iter
print("Standard")
print(list(E_ML))
print("U")
print(list(E_ML_U))
print("Standard iter")
print(niter_machinelearn_dist)
print("U iter")
print(niter_machinelearn_U)
sys.exit(1)
# This "proves" that the machine learning approach works approximately as good as the sum approximation, as it gives us an "amazing" starting guess.
# The number of necessary CCSD calculations is then pretty low.
outdata={}
outdata["basis"]=basis
outdata["molecule_name"]=molecule_name
outdata["sample_geometries"]=sample_geom1
outdata["test_geometries"]=geom_alphas1
outdata["sample_energies"]=sample_energies
outdata["GP"]=niter_machinelearn_guess

outdata["energies_ML"]=E_ML
file="energy_data/convergence_%s_%s_%d_new.bin"%(molecule_name,basis,len(sample_geom1))
import pickle
with open(file,"wb") as f:
    pickle.dump(outdata,f)
sys.exit(1)
