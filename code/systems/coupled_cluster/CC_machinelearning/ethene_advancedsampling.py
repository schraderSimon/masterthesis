"""
Get approximate AMP-CCEVC amplitudes for HF molecule in small basis
"""
import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from machinelearning import *
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVDZ'
#basis="6-31G*"
molecule_name="ethene"
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
def molecule(x):
    C_pos=2.482945+x
    H_pos=3.548545+x
    return "C 0 0 0 ;C 0 0 %f;H 0 1.728121 -1.0656 ;H 0 -1.728121 -1.0656 ;H 0 1.728121 %f;H 0 -1.728121 %f"%(C_pos,H_pos,H_pos)
refx=[0]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geom1=np.linspace(-0.9,2.7,2)
num_samples=2
max_samples=7
sample_indices=[0,80]
geom_alphas1=np.linspace(-1,2.8,77)
target_U=get_U_matrix(geom_alphas1,molecule,basis,reference_determinant)
target_U_sampling=target_U[3:-3]
sample_U=get_U_matrix(sample_geom1,molecule,basis,reference_determinant)
geom_alphas_sampling1=geom_alphas1[3:-3]
import pickle
geom_alphas=[[x] for x in geom_alphas1]
geom_alphas_sampling=[[x] for x in geom_alphas_sampling1]
sample_geom=[[x] for x in sample_geom1]

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")

while num_samples < max_samples: #As long as I want to add samples:
    evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)

    """
    Set up machine learning for t amplitudes
    """
    kernel=RBF_kernel_unitary_matrices #Use standard RBF kernel
    stds=np.ones(len(geom_alphas_sampling1))
    predictions=[]

    """
    Set up machine learning for t amplitudes
    """
    t1s_orth,t2s_orth,t_coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)
    for i in range(len(sample_geom)):
        mean,std=get_model(sample_U,t_coefs[i]-np.mean(t_coefs[i]),kernel,target_U_sampling)
        predictions.append(mean+np.mean(t_coefs[i]))
        stds+=(std)
    largest_std_pos=np.argmax(stds) #The position with the largest std
    sample_geom.append(geom_alphas_sampling[largest_std_pos])
    sample_geom1=list(sample_geom1)
    sample_geom1.append(geom_alphas_sampling1[largest_std_pos])
    sample_geom1=np.array(sample_geom1)
    sample_U.append(target_U_sampling[largest_std_pos])
    newt1,newt2,newl1,newl2,nwesample_energies=setUpsamples([sample_geom[-1]],molecule,basis,reference_determinant,mix_states=False,type="procrustes")
    t1s.append(newt1)
    t2s.append(newt2)
    l1s.append(newt1)
    l2s.append(newt2)
    print(stds)
    print(sample_geom)
    num_samples+=1
evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)

"""
Set up machine learning for t amplitudes
"""
kernel=RBF_kernel_unitary_matrices #Use standard RBF kernel
stds=np.ones(len(geom_alphas_sampling1))
predictions=[]

"""
Set up machine learning for t amplitudes
"""
t1s_orth,t2s_orth,t_coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)
for i in range(len(sample_geom)):
    mean,std=get_model(sample_U,t_coefs[i]-np.mean(t_coefs[i]),kernel,target_U)
    predictions.append(mean+np.mean(t_coefs[i]))

t1s_orth,t2s_orth,t_coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)
t1_machinelearn=[]
t2_machinelearn=[]
means=np.array(predictions)
for i in range(len(geom_alphas1)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(t_coefs)):
        t1_temp+=means[j][i]*t1s_orth[j]
        t2_temp+=means[j][i]*t2s_orth[j]
    t1_machinelearn.append(t1_temp)
    t2_machinelearn.append(t2_temp)

print("Initial")
xtol=1e-8 #Convergence tolerance
E_ML_U=evcsolver.calculate_CCSD_energies_from_guess(t1_machinelearn,t2_machinelearn,xtol=xtol)

evcsolver.solve_CCSD_startguess(t1_machinelearn,t2_machinelearn,xtol=xtol)
niter_machinelearn=evcsolver.num_iter
outdata={}
outdata["basis"]=basis
outdata["molecule_name"]=molecule_name
outdata["sample_geometries"]=sample_geom1
outdata["test_geometries"]=geom_alphas1
outdata["sample_energies"]=sample_energies
outdata["ML"]=niter_machinelearn

outdata["coefficients"]=t_coefs
outdata["sample_U"]=sample_U
outdata["target_U"]=target_U
outdata["CC_sample_amplitudes_procrustes"]=[t1s_orth,t2s_orth]
outdata["CC_sample_amplitudes"]=[t1s,t2s,l1s,l2s]
outdata["reference_determinant"]=reference_determinant

outdata["energies_ML"]=E_ML_U
file="energy_data/ethene_machinelearning_bestGeometries_%s_%d.bin"%(basis,len(sample_geom1))
import pickle
with open(file,"wb") as f:
    pickle.dump(outdata,f)
sys.exit(1)
