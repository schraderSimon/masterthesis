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
num_points=25
basis=basis
file="energy_data/BeH2_machinelearning_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML=pickle.load(f)
file="energy_data/BeH2_machinelearning_bestGeometries_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML_top=pickle.load(f)
molecule_name="BeH2_asymmetric"
basis="cc-pVDZ"
len_sample_geom=16
type="U"
#type="avstand"
file="../energy_data/convergence_%s2D_%s_%d.bin"%(molecule_name,basis,len_sample_geom)
import pickle
with open(file,"rb") as f:
    energy_dict=pickle.load(f)
#std_average=np.mean(energy_dict["std"],axis=0)
#print(std_average.shape)
#std_average=std_average.reshape((10,10))
CCSD=np.array(energy_dict["E_CCSD"]).reshape((10,10))
E_ML=np.array(data_ML["energies_ML"]).reshape((10,10))
E_ML_appr=np.array(data_ML_top["energies_ML"]).reshape((10,10))
print(CCSD)
print(E_ML)
print(E_ML_appr)
x=y=np.linspace(2,6,10)
test_geom=energy_dict["test_geometries"]
E_MLerr=(E_ML-CCSD)*1000
E_ML_appr_err=(E_ML_appr-CCSD)*1000
cmap="jet"
z_min=np.amin( np.concatenate( (E_MLerr.ravel(),E_ML_appr_err.ravel())))
alpha=1
z_max=np.amax( np.concatenate( (E_MLerr.ravel(),E_ML_appr_err.ravel())))
#z_max=np.amax(E_MLerr.ravel())
fig,grid=plt.subplots(1,2,sharey=True,sharex=True,figsize=(20,6))
im0=grid[0].pcolormesh(x, y, E_MLerr, cmap=cmap,shading='auto',vmin=z_min,vmax=z_max,alpha=alpha)
grid[0].set_title("ML ")
#grid[0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[0].set_ylabel(r"distance $H^1$-Be (Bohr)")

im1=grid[1].pcolormesh(x, y, E_ML_appr_err, cmap=cmap,shading='auto',vmin=z_min,vmax=z_max,alpha=alpha)
grid[1].set_title("ML impro")
grid[1].set_xlabel(r"distance $H^2$-Be (Bohr)")
colorbar=fig.colorbar(im0,label='Error (mHartree)')

plt.legend()
plt.show()
