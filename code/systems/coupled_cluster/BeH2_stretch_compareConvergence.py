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
x=np.linspace(2.1,5.9,5)
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
energy_dict={}
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
t1s_orth,t2s_orth,t_coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)
t1_machinelearn=[]
t2_machinelearn=[]
means_U=[]; std_U=[];
means_avstand=[]; std_avstand=[];
sample_U=get_U_matrix(sample_geom,molecule,basis,reference_determinant)
target_U=get_U_matrix(geom_alphas,molecule,basis,reference_determinant)
sample_C=get_coulomb_matrix(sample_geom,molecule)
target_C=get_coulomb_matrix(geom_alphas,molecule)
for i in range(len(sample_geom)):
    mean_U,std=multivariate_gaussian_gpy_matrixInput(sample_U,t_coefs[i],target_U,sigma=1,l=1)
    means_U.append(mean_U)
    std_U.append(std)
    mean_avstand,std=multivariate_gaussian_gpy(sample_geom,t_coefs[i],geom_alphas,sigma=1,l=1)
    means_avstand.append(mean_avstand)
    std_avstand.append(std)
means_U=np.array(means_U)
print(mean_U)
sys.exit(1)
means_avstand=np.array(means_avstand)
std_U=np.array(std_U)
std_avstand=np.array(std_avstand)

type="U"
if type=="U":
    means=means_U
    standardDeviation=std_U

elif type=="avstand":
    means=means_avstand
    standardDeviation=std_avstand
    print("Type is avstand")
else:
    sys.exit(1)
for i in range(len(means_U)):
    print(means_U[i],means_U[i]-means_avstand[i])
for i in range(len(geom_alphas)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(t_coefs)):
        t1_temp+=means[j][i]*t1s_orth[j]
        t2_temp+=means[j][i]*t2s_orth[j]
    t1_machinelearn.append(t1_temp)
    t2_machinelearn.append(t2_temp)
xtol=1e-8 #Convergence tolerance

outdata={}

#E_CCSD=evcsolver.solve_CCSD_noProcrustes(xtol=xtol)
#niter_CCSD=evcsolver.num_iter
#evcsolver.solve_CCSD_startguess(t1_machinelearn,t2_machinelearn,xtol=xtol)
#niter_machinelearn_guess=evcsolver.num_iter
E_ML=evcsolver.calculate_CCSD_energies_from_guess(t1_machinelearn,t2_machinelearn,xtol=xtol)
outdata["E_machineLearn"]=E_ML
file="energy_data/Coulomb_test.bin"
#file="energy_data/convergence_%s2D_%s_%d_%s.bin"%(molecule_name,basis,len(sample_geom),type)
import pickle
with open(file,"wb") as f:
    pickle.dump(outdata,f)
sys.exit(1)

E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=0.3,xtol=1e-5)
t1s_reduced=evcsolver.t1s_final
t2s_reduced=evcsolver.t2s_final
evcsolver.solve_CCSD_startguess(t1s_reduced,t2s_reduced,xtol=xtol)
niter_AMP_startguess=evcsolver.num_iter
niter_machinelearn_guess=np.array(niter_machinelearn_guess)
niter_AMP_startguess=np.array(niter_AMP_startguess)
niter_CCSD=np.array(niter_CCSD)
print(niter_machinelearn_guess)
print(niter_AMP_startguess)
print(niter_CCSD)
print("Machine learning: %.1f, MP2: %.1f, AMP: %.1f"%(np.mean(niter_machinelearn_guess),np.mean(niter_CCSD),np.mean(niter_AMP_startguess)))
outdata["basis"]=basis
molecule_name="BeH2_asymmetric"
outdata["molecule_name"]=molecule_name
outdata["sample_geometries"]=sample_geom
outdata["test_geometries"]=geom_alphas
outdata["MP2"]=niter_CCSD
outdata["EVC"]=niter_AMP_startguess
outdata["GP"]=niter_machinelearn_guess
outdata["E_CCSD"]=E_CCSD
outdata["E_AMP_red"]=E_AMP_red
outdata["std"]=standardDeviation
outdata["mean"]=means
