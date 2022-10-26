import sys
sys.path.append("../libraries")
from rccsd_gs import *
from func_lib import *
from numba import jit
from machinelearning import *
from matrix_operations import *
from helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid

basis = 'cc-pVDZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
def molecule(x,y):
    return """Be 0 0 0; H -%f 0 0; H %f 0 0"""%(x,y)
refx=(2,2)
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
n=n0=20
x=4*np.random.rand(n,2)+2 #n random numbers between 2 and 6 for x and y directions
sample_geom_new=[]
x=np.linspace(2.1,5.9,4)
for i in range(len(x)):
    for j in range(len(x)):
        sample_geom_new.append([x[i],x[j]])

n=n0
sample_geom=np.array(sample_geom_new)
print(len(sample_geom))
span=np.linspace(2,6,10)
geom_alphas=[]
for x in span:
    for y in span:
        geom_alphas.append((x,y))
x, y = np.meshgrid(span,span)


outdata={}

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
t1s_orth,t2s_orth,t_coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)

t1_machinelearn=[]
t2_machinelearn=[]
means_U=[]; std_U=[];
means_avstand=[]; std_avstand=[];
sample_U=get_U_matrix(sample_geom,molecule,basis,reference_determinant)
target_U=get_U_matrix(geom_alphas,molecule,basis,reference_determinant)
outdata["coefficients"]=t_coefs
outdata["sample_U"]=sample_U
outdata["target_U"]=target_U
outdata["sample_geometries"]=sample_geom
outdata["target_geometries"]=geom_alphas
outdata["CC_sample_amplitudes_procrustes"]=[t1s_orth,t2s_orth]
outdata["CC_sample_amplitudes"]=[t1s,t2s,l1s,l2s]
outdata["basis_set"]=basis_set
outdata["reference_determinant"]=reference_determinant
file="energy_data/GP_input_data_BeH2_%d.bin"%(len(sample_geom))
#file="energy_data/convergence_%s2D_%s_%d_%s.bin"%(molecule_name,basis,len(sample_geom),type)
import pickle
with open(file,"wb") as f:
    pickle.dump(outdata,f)
sys.exit(1)

for i in range(len(sample_geom)):
    mean_U,std=multivariate_gaussian_gpy_matrixInput(sample_U,t_coefs[i],target_U,sigma=1,l=1)
    means_U.append(mean_U)
    std_U.append(std)
    mean_avstand,std=multivariate_gaussian_gpy(sample_C,t_coefs[i],target_C,sigma=1,l=1)
    means_avstand.append(mean_avstand)
    std_avstand.append(std)
means_U=np.array(means_U)
print(mean_U)
sys.exit(1)
means_avstand=np.array(means_avstand)
std_U=np.array(std_U)
std_avstand=np.array(std_avstand)
standardDeviation=std_avstand

t1_machinelearn=[]
t2_machinelearn=[]
for i in range(len(geom_alphas)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(t_coefs)):
        t1_temp+=means_U[j][i]*t1s_orth[j]
        t2_temp+=means_U[j][i]*t2s_orth[j]
    t1_machinelearn.append(t1_temp)
    t2_machinelearn.append(t2_temp)
xtol=1e-8 #Convergence tolerance
E_ML_U=evcsolver.calculate_CCSD_energies_from_guess(t1_machinelearn,t2_machinelearn,xtol=xtol)
t1_machinelearn=[]
t2_machinelearn=[]
for i in range(len(geom_alphas)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(t_coefs)):
        t1_temp+=means_avstand[j][i]*t1s_orth[j]
        t2_temp+=means_avstand[j][i]*t2s_orth[j]
    t1_machinelearn.append(t1_temp)
    t2_machinelearn.append(t2_temp)
xtol=1e-8 #Convergence tolerance
E_ML_dist=evcsolver.calculate_CCSD_energies_from_guess(t1_machinelearn,t2_machinelearn,xtol=xtol)
