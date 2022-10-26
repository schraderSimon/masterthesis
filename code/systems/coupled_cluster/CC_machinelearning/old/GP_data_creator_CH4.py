import sys
sys.path.append("../../libraries")
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
refx=[1]
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
n=n0=20
x=4*np.random.rand(n,2)+2 #n random numbers between 2 and 6 for x and y directions
sample_geom1=np.linspace(0.8,3,7)
sample_geom=[[x] for x in sample_geom1]
geom_alphas1=np.linspace(0.7,3.3,60)
geom_alphas=[[x] for x in geom_alphas1]


outdata={}
outdata["CCSD energy"]=CCSD_energy_curve(molecule,geom_alphas,basis)
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
file="energy_data/GP_input_data_CH4_%d.bin"%(len(sample_geom))
#file="energy_data/convergence_%s2D_%s_%d_%s.bin"%(molecule_name,basis,len(sample_geom),type)
import pickle
with open(file,"wb") as f:
    pickle.dump(outdata,f)
sys.exit(1)
