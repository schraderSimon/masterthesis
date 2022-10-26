#Required: Different values for the coefficients
#Required: Unitary matrices U
#Required: Exact (CC) energies
#Required: CC energy equation
import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from func_lib import *
from matrix_operations import *
from helper_functions import *
import pickle
def molecule(x,y):
    return """Be 0 0 0; H -%f 0 0; H %f 0 0"""%(x,y)
len_sample_geom=16
molecule_name="BeH2_asymmetric"

def molecule(x):
    pos=np.array([[ 0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  1.089  ],
       [ 1.02672,  0.     , -0.363  ],
       [-0.51336,  0.88916, -0.363  ],
       [-0.51336, -0.88916, -0.363  ]])*x
    types=["C","H","H","H","H"]
    string=""
    for i in range(len(types)):
        string+="%s %f %f %f;"%(types[i],pos[i,0],pos[i,1],pos[i,2])
    return string
len_sample_geom=7
molecule_name="CH4"
basis="cc-pVDZ"
"""
file="energy_data/convergence_%s2D_%s_%d.bin"%(molecule_name,basis,len_sample_geom)
with open(file,"rb") as f:
    energy_dict=pickle.load(f)
"""



file="energy_data/GP_input_data_%s_%d.bin"%(molecule_name,len_sample_geom)
from scipy.optimize import minimize, minimize_scalar

with open(file,"rb") as f:
    data=pickle.load(f)
coefs=data["coefficients"]
sample_U=data["sample_U"]
target_U=data["target_U"]
sample_geom=data["sample_geometries"]
geom_alphas=data["target_geometries"]
[t1s_orth,t2s_orth]=data["CC_sample_amplitudes_procrustes"]
[t1s,t2s,l1s,l2s]=data["CC_sample_amplitudes"]
basis_set=data["basis_set"]
reference_determinant=data["reference_determinant"]
CCSD=np.array(data["CCSD energy"])#.reshape((10,10))

predictions=[]
stds=np.zeros(len(geom_alphas))
for i in range(len(sample_geom)):

    y=coefs[i]
    mean_y=np.mean(y)
    predict,std=get_model(sample_U,y-np.mean(y),RBF_kernel_unitary_matrices,target_U)
    predictions.append(predict+mean_y)
    stds+=(std)


("Done predicting")
t1_machinelearn=[]
t2_machinelearn=[]
for i in range(len(geom_alphas)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(coefs)):
        t1_temp+=predictions[j][i]*t1s_orth[j]
        t2_temp+=predictions[j][i]*t2s_orth[j]
    t1_machinelearn.append(t1_temp)
    t2_machinelearn.append(t2_temp)
evcsolver=EVCSolver(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
E_ML_U=evcsolver.calculate_CCSD_energies_from_guess(t1_machinelearn,t2_machinelearn,xtol=1e-8)
E_ML_U=np.array(E_ML_U)#.reshape((10,10))
print(1000*(E_ML_U-CCSD))
stds=1000*stds#.reshape((10,10))
print("Standard deviations:")
print(stds)
print(1000*(E_ML_U-CCSD)/stds)
