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
from matplotlib.ticker import MaxNLocator
num_points=7
basis=basis
file="energy_data/ethene_machinelearning_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML=pickle.load(f)
file="energy_data/ethene_machinelearning_bestGeometries_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML_top=pickle.load(f)
file="energy_data/ethene_AMPCCEVC_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_AMP=pickle.load(f)
CCSD_energies=np.array(data_AMP["energies_CCSD"])
niter_AMP_startguess10_7=data_AMP["EVC_10"]
niter_AMP_startguess20_7=data_AMP["EVC_20"]
niter_prevGeom=data_AMP["prevGeom"]
niter_ML_7=data_ML["ML"]
niter_ML_top_7=data_ML_top["ML"]
niter_MP2=data_AMP["MP2"]
AMP_20_energies_7=data_AMP["energies_AMP_20"]
AMP_10_energies_7=data_AMP["energies_AMP_10"]
ML_energies_7=np.array(data_ML["energies_ML"])
sample_geom1_7=np.array(data_ML["sample_geometries"])+2.482945

ML_top_energies_7=np.array(data_ML_top["energies_ML"])
sample_geom2_7=np.array(data_ML_top["sample_geometries"])+2.482945

num_points=10
file="energy_data/ethene_machinelearning_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML=pickle.load(f)
file="energy_data/ethene_machinelearning_bestGeometries_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML_top=pickle.load(f)
file="energy_data/ethene_AMPCCEVC_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_AMP=pickle.load(f)
niter_AMP_startguess10_10=data_AMP["EVC_10"]
niter_AMP_startguess20_10=data_AMP["EVC_20"]
niter_prevGeom=data_AMP["prevGeom"]
niter_ML_10=data_ML["ML"]
niter_ML_top_10=data_ML_top["ML"]
niter_MP2=data_AMP["MP2"]
AMP_20_energies_10=data_AMP["energies_AMP_20"]
AMP_10_energies_10=data_AMP["energies_AMP_10"]
ML_energies_10=np.array(data_ML["energies_ML"])
geom_alphas1=np.array(data_ML["test_geometries"])+2.482945
sample_geom1_10=np.array(data_ML["sample_geometries"])+2.482945

ML_top_energies_10=np.array(data_ML_top["energies_ML"])
sample_geom2_10=np.array(data_ML_top["sample_geometries"])+2.482945


fig,ax=plt.subplots(2,1,sharey=False,sharex=True,figsize=(10,10))
for x in (sample_geom1_7[:-1]):
    ax[0].axvline(x,linestyle="--",color="blue",alpha=0.3)
ax[0].axvline(sample_geom1_7[-1],linestyle="--",color="blue",alpha=0.3,label="Sample geom.")
for x in (sample_geom2_7[:-1]):
    ax[0].axvline(x,linestyle="--",color="red",alpha=0.3)
ax[0].axvline(sample_geom2_7[-1],linestyle="--",color="red",alpha=0.3, label="Sample geom. (auto)")

ax[0].plot(geom_alphas1,1000*(CCSD_energies-ML_energies_7),label="GP",color="blue")
ax[0].plot(geom_alphas1,1000*(CCSD_energies-ML_top_energies_7),color="red",label="GP (auto)")
ax[0].plot(geom_alphas1,1000*(CCSD_energies-AMP_10_energies_7),label=r"tr. sum 10%",color="green")
#ax[0].plot(geom_alphas1,1000*(CCSD_energies-AMP_20_energies_7),label=r"tr. sum 20%",color="darkorange")
ax[0].fill_between(geom_alphas1,-1.6,1.6,color="aqua",alpha=0.3,label="Chemical accuracy")
ax[0].set_ylabel(r"$\Delta E$ (mHartree)")
for x in (sample_geom1_10):
    ax[1].axvline(x,linestyle="--",color="blue",alpha=0.3)
for x in (sample_geom2_10):
    ax[1].axvline(x,linestyle="--",color="red",alpha=0.3)

ax[1].plot(geom_alphas1,1000*(CCSD_energies-ML_energies_10),label="GP",color="blue")
ax[1].plot(geom_alphas1,1000*(CCSD_energies-AMP_10_energies_10),label=r"tr. sum 10%",color="green")
#ax[1].plot(geom_alphas1,1000*(CCSD_energies-AMP_20_energies_10),label=r"tr. sum 20%")
ax[1].plot(geom_alphas1,1000*(CCSD_energies-ML_top_energies_10),label="GP (auto)",color="red")

ax[1].set_ylabel(r"$\Delta E$ (mHartree)")
ax[1].set_xlabel("C-C distance (Bohr)")
ax[0].legend(columnspacing=0.0,handletextpad=0.0,labelspacing=0)
ax[1].set_title("10 sample geometries")
ax[0].set_title("7 sample geometries")
ax[0].set_ylim([-3,6])
ax[1].set_ylim([-0.2,0.2])
plt.suptitle("Deviation from CCSD energy (ethylene)")

plt.tight_layout()
plt.savefig("plots/ethene_energy.pdf")
plt.show()

fig,ax=plt.subplots(2,1,sharey=False,sharex=True,figsize=(10,10))

for x in (sample_geom1_7[:-1]):
    ax[0].axvline(x,linestyle="--",color="blue",alpha=0.3)
ax[0].axvline(sample_geom1_7[-1],linestyle="--",color="blue",alpha=0.3,label="Sample geom.")
for x in (sample_geom2_7[:-1]):
    ax[0].axvline(x,linestyle="--",color="red",alpha=0.3)
ax[0].axvline(sample_geom2_7[-1],linestyle="--",color="red",alpha=0.3, label="Sample geom. (auto)")

ax[0].plot(geom_alphas1,niter_ML_7,label="GP",color="blue")
ax[0].plot(geom_alphas1,niter_ML_top_7,label="GP (auto)",color="red")

ax[0].plot(geom_alphas1,niter_AMP_startguess10_7,label=r"tr. sum 10%",color="green")
#ax[0].plot(geom_alphas1,niter_AMP_startguess20_7,label=r"tr. sum 20%",color="darkorange")
ax[0].plot(geom_alphas1,niter_MP2,label="MP2",color="purple")
ax[0].plot(geom_alphas1,niter_prevGeom,label="prevGeom",color="brown")

ax[0].set_title("7 sample geometries")
ax[0].set_ylabel(r"Number of iterations")
for x in (sample_geom1_10):
    ax[1].axvline(x,linestyle="--",color="blue",alpha=0.3)
for x in (sample_geom2_10):
    ax[1].axvline(x,linestyle="--",color="red",alpha=0.3)

ax[1].plot(geom_alphas1,niter_ML_10,label="GP",color="blue")
ax[1].plot(geom_alphas1,niter_ML_top_10,label="GP (auto)",color="red")

ax[1].plot(geom_alphas1,niter_AMP_startguess10_10,label=r"tr. sum 10%",color="green")
#ax[1].plot(geom_alphas1,niter_AMP_startguess20_10,label=r"tr. sum 20%",color="darkorange")
ax[1].plot(geom_alphas1,niter_MP2,label="MP2",color="purple")
ax[1].plot(geom_alphas1,niter_prevGeom,label="prevGeom",color="brown")

ax[1].set_ylim([6,21])

ax[0].set_ylim([6,21])
ax[1].set_ylabel(r"Number of iterations")
ax[1].set_xlabel("C-C distance (Bohr)")
ax[1].set_title("10 sample geometries")

ax[0].legend(loc="upper left",columnspacing=0.0,handletextpad=0.0,labelspacing=0)
plt.suptitle("Number of iterations (ethylene)")
#ax[0].set_yticks(np.linspace(6,22,9,dtype=int))
#ax[1].set_yticks(np.linspace(6,22,9,dtype=int))
for i in range(len(ax)):
    ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("plots/ethene_niter.pdf")

plt.show()
