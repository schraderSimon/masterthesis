import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from machinelearning import *
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVDZ'
import pickle
num_points=7
basis=basis
file="energy_data/ethene_machinelearning_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML=pickle.load(f)
file="energy_data/ethene_AMPCCEVC_%s_%d.bin"%(basis,7)
with open(file,"rb") as f:
    data_AMP=pickle.load(f)
CCSD_energies=np.array(data_AMP["energies_CCSD"])
ML_energies=np.array(data_ML["energies_ML"])
geom_alphas1=np.array(data_ML["test_geometries"])
sample_geom=np.array(data_ML["sample_geometries"])
t_coefs=data_ML["coefficients"]
sample_U=data_ML["sample_U"]
target_U=data_ML["target_U"]
t1s_orth,t2s_orth=data_ML["CC_sample_amplitudes_procrustes"]
t1s,t2s,l1s,l2s=data_ML["CC_sample_amplitudes"]
from scipy.stats import ortho_group

#t_coefs=t_coefs@ortho_group.rvs(len(t_coefs))

plt.plot(sample_geom[:],t_coefs[:,:])
kernel=extended_RBF_kernel_unitary_matrices #Use standard RBF kernel
stds=np.zeros(len(geom_alphas1))
predictions=[]
for i in range(len(sample_geom)):
    mean,std=get_model(sample_U,(t_coefs.T[i]-np.mean(t_coefs[i])),kernel,target_U)
    predictions.append(mean+np.mean(t_coefs[i]))
    stds+=(std)
means=np.array(predictions)
plt.plot(geom_alphas1,means.T[:,])
for x in sample_geom:
    plt.axvline(x)
plt.show()

from scipy import interpolate
CCSD_energy_func= interpolate.interp1d(geom_alphas1, CCSD_energies)
plt.plot(geom_alphas1,1000*(CCSD_energies-ML_energies),label="ML")
for x in sample_geom:
    plt.axvline(x)

plt.legend()
plt.show()
