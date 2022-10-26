import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from machinelearning import *
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVTZ'
import pickle
num_points=7
basis=basis
file="energy_data/HF_machinelearning_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML=pickle.load(f)
file="energy_data/HF_machinelearning_bestGeometries_%s_%d.bin"%(basis,num_points+1)
with open(file,"rb") as f:
    data_ML_top=pickle.load(f)
file="energy_data/HF_AMPCCEVC_%s_%d.bin"%(basis,num_points+3)
with open(file,"rb") as f:
    data_AMP=pickle.load(f)
CCSD_energies=np.array(data_AMP["energies_CCSD"])
ML_energies=np.array(data_ML["energies_ML"])
geom_alphas1=np.array(data_ML["test_geometries"])
ML_top_energies=np.array(data_ML_top["energies_ML"])
geom_alphas2=np.array(data_ML_top["test_geometries"])
from scipy import interpolate
CCSD_energy_func= interpolate.interp1d(geom_alphas1, CCSD_energies)
plt.plot(geom_alphas1,1000*(CCSD_energies-ML_energies),label="ML")
plt.plot(geom_alphas2,1000*(CCSD_energy_func(geom_alphas2)-ML_top_energies),label="ML (improved)")

plt.legend()
plt.show()
