#Required: Different values for the coefficients
#Required: Unitary matrices U
#Required: Exact (CC) energies
#Required: CC energy equation
import sys
sys.path.append("../libraries")
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

def RBF_kernel_unitary_matrices(list_U1,list_U2,kernel_params=[1,1,1]):
    sigma_1=np.exp(kernel_params[0])
    l_1=np.exp(kernel_params[1])
    sigma_2=np.exp(kernel_params[2])
    l_2=np.exp(kernel_params[3])
    noise=0
    norm=np.zeros((len(list_U1),len(list_U2)))
    n=min([len(list_U1),len(list_U2)])
    for i in range(len(list_U1)):
        for j in range(len(list_U2)):
            U1=list_U1[i]
            U2=list_U2[j]
            norm[i,j]=np.linalg.norm(U1-U2)
    #kernel_mat= sigma*np.exp(-0.5*norm**2/l)
    kernel_mat= sigma_1+np.exp(-0.5*norm**2/l_1)+sigma_2*np.exp(-0.5*norm/l_2)
    for i in range(n):
        kernel_mat[i,i]+=noise
    return kernel_mat
def RQ_kernel_unitary_matrices(list_U1,list_U2,kernel_params=[1,1,100]):
    sigma=np.exp(kernel_params[0])
    l=np.exp(kernel_params[1])
    alpha=np.exp(kernel_params[2])
    norm=np.zeros((len(list_U1),len(list_U2)))
    for i in range(len(list_U1)):
        for j in range(len(list_U2)):
            U1=list_U1[i]
            U2=list_U2[j]
            norm[i,j]=np.linalg.norm(U1-U2)
    kernel_mat=sigma**2*(np.ones_like(norm)+0.5*norm**2/(alpha*l**2))**(-alpha)
    return kernel_mat

def GP(X1, y1, X2, kernel_func,kernel_params):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1,kernel_params)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2,kernel_params)
    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2,kernel_params)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance
def log_likelihood(kernel_params,data_X,y,kernel):
    cov_matrix=kernel(data_X,data_X,kernel_params) #The covariance matrix
    det=np.linalg.det(cov_matrix)
    log_det=np.log(det)
    #inverse=np.linalg.inv(cov_matrix+1e-10*np.eye(len(cov_matrix)))
    inv_times_data=np.linalg.solve(cov_matrix,y)
    return -(-0.5*y.T@inv_times_data-0.5*log_det)
def find_best_model(U_list,y,kernel):
    y_new=y
    sol=minimize(log_likelihood,x0=[0,0,0,0],args=(U_list,y,kernel))
    best_sigma=sol.x
    print(best_sigma,sol.fun)
    #print(kernel(U_list,U_list,sol.x))
    return best_sigma
def get_model(U_list,y,kernel,U_list_target):
    best_sigma=find_best_model(U_list,y,kernel)
    new, Σ2 = GP(U_list, y, U_list_target, kernel,best_sigma)
    return new, np.diag(Σ2)



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
