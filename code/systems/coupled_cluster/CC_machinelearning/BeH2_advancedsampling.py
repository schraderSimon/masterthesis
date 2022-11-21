import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from func_lib import *
from numba import jit
from machinelearning import *
from matrix_operations import *
from helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid

molecule_name="BeH2"

basis = 'cc-pVTZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
def molecule(x,y):
    return """Be 0 0 0; H -%f 0 0; H %f 0 0"""%(x,y)
refx=(2,2)
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
n=n0=20
x=4*np.random.rand(n,2)+2 #n random numbers between 2 and 6 for x and y directions
span=np.linspace(2,6,10)
geom_alphas=[]
for x in span:
    for y in span:
        geom_alphas.append((x,y))
sample_geom=[(2.1,2.1),(2.1,5.9),(5.9,2.1),(5.9,5.9)]
x, y = np.meshgrid(span,span)

target_U=get_U_matrix(geom_alphas,molecule,basis,reference_determinant)

"""
Set up machine learning for t amplitudes
"""
num_samples=4
max_samples=30
sample_U=get_U_matrix(sample_geom,molecule,basis,reference_determinant)
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")

while num_samples < max_samples: #As long as I want to add samples:

    evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)

    """
    Set up machine learning for t amplitudes
    """
    t1s_orth,t2s_orth,t_coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)

    kernel=RBF_kernel_unitary_matrices #Use standard RBF kernel
    stds=np.ones(len(geom_alphas))
    predictions=[]
    for i in range(len(sample_geom)):
        mean,std=get_model(sample_U,t_coefs[i]-np.mean(t_coefs[i]),kernel,target_U)
        predictions.append(mean+np.mean(t_coefs[i]))
        stds+=(std)
    largest_std_pos=np.argmax(stds) #The position with the largest std
    sample_geom.append(geom_alphas[largest_std_pos])
    sample_U.append(target_U[largest_std_pos])
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
t1s_orth,t2s_orth,t_coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)

kernel=RBF_kernel_unitary_matrices #Use standard RBF kernel
stds=np.ones(len(geom_alphas))
predictions=[]
for i in range(len(sample_geom)):
    mean,std=get_model(sample_U,t_coefs[i]-np.mean(t_coefs[i]),kernel,target_U)
    predictions.append(mean+np.mean(t_coefs[i]))
    stds+=(std)
t1s_orth,t2s_orth,t_coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)
t1_machinelearn=[]
t2_machinelearn=[]
kernel=RBF_kernel_unitary_matrices #Use extended RBF kernel
stds=np.zeros(len(geom_alphas))
predictions=[]

for i in range(len(sample_geom)):
    mean,std=get_model(sample_U,t_coefs[i]-np.mean(t_coefs[i]),kernel,target_U)
    predictions.append(mean+np.mean(t_coefs[i]))
    stds*=(std)

means=np.array(predictions)
for i in range(len(geom_alphas)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(t_coefs)):
        t1_temp+=means[j][i]*t1s_orth[j]
        t2_temp+=means[j][i]*t2s_orth[j]
    t1_machinelearn.append(t1_temp)
    t2_machinelearn.append(t2_temp)

print("Initial")
xtol=1e-5 #Convergence tolerance
E_ML_U=evcsolver.calculate_CCSD_energies_from_guess(t1_machinelearn,t2_machinelearn,xtol=xtol)

evcsolver.solve_CCSD_startguess(t1_machinelearn,t2_machinelearn,xtol=xtol)
niter_machinelearn=evcsolver.num_iter
outdata={}
outdata["basis"]=basis
outdata["molecule_name"]=molecule_name
outdata["sample_geometries"]=sample_geom
outdata["test_geometries"]=geom_alphas
outdata["sample_energies"]=sample_energies
outdata["ML"]=niter_machinelearn

outdata["coefficients"]=t_coefs
outdata["sample_U"]=sample_U
outdata["target_U"]=target_U
outdata["CC_sample_amplitudes_procrustes"]=[t1s_orth,t2s_orth]
outdata["CC_sample_amplitudes"]=[t1s,t2s,l1s,l2s]
outdata["reference_determinant"]=reference_determinant

outdata["energies_ML"]=E_ML_U
file="energy_data/BeH2_machinelearning_bestGeometries_%s_%d.bin"%(basis,len(sample_geom))
import pickle
with open(file,"wb") as f:
    pickle.dump(outdata,f)
sys.exit(1)
