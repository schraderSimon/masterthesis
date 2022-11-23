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

import matplotlib as mpl
mpl.rcParams['font.size'] = 25

num_points=25
basis=basis
file="energy_data/BeH2_machinelearning_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML=pickle.load(f)
file="energy_data/BeH2_machinelearning_bestGeometries_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML_top=pickle.load(f)
file="energy_data/BeH2_AMPCCEVC_%s_%d.bin"%(basis,num_points)

import pickle
with open(file,"rb") as f:
    energy_dict=pickle.load(f)
ML_sample_geometries=np.array(data_ML["sample_geometries"])
ML_top_sample_geometries=np.array(data_ML_top["sample_geometries"])
CCSD=np.array(energy_dict["energies_CCSD"]).reshape((10,10))
E_ML=np.array(data_ML["energies_ML"]).reshape((10,10))
E_ML_auto=np.array(data_ML_top["energies_ML"]).reshape((10,10))
E_AMPCCEVC_20=np.array(energy_dict["energies_AMP_20"]).reshape((10,10))
E_AMPCCEVC_10=np.array(energy_dict["energies_AMP_10"]).reshape((10,10))

niter_AMP_10=np.array(energy_dict["EVC_10"]).reshape((10,10))
niter_AMP_20=np.array(energy_dict["EVC_20"]).reshape((10,10))
niter_prevGeom=np.array(energy_dict["prevGeom"]).reshape((10,10))
niter_ML=np.array(data_ML["ML"]).reshape((10,10))
niter_ML_auto=np.array(data_ML_top["ML"]).reshape((10,10))
niter_MP2=np.array(energy_dict["MP2"]).reshape((10,10))


print(CCSD)
print(E_ML)
print(E_ML_auto)
x=y=np.linspace(2,6,10)
test_geom=energy_dict["test_geometries"]
E_MLerr=abs(E_ML-CCSD)*1000
E_ML_auto_err=abs(E_ML_auto-CCSD)*1000
E_20_err=abs(E_AMPCCEVC_20-CCSD)*1000
E_10_err=abs(E_AMPCCEVC_10-CCSD)*1000

cmap="jet"
z_min=0
alpha=0.9
z_max=np.amax( np.concatenate( (E_MLerr.ravel(),E_ML_auto_err.ravel())))
#z_max=np.amax(E_MLerr.ravel())
fig,grid=plt.subplots(2,2,sharey=True,sharex=True,figsize=(15,10))
im0=grid[0,0].pcolormesh(x, y, E_MLerr, cmap=cmap,shading='auto',vmin=z_min,vmax=np.amax(E_MLerr),alpha=alpha)
grid[0,0].set_title("GP")
grid[0,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[0,0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0,0].scatter(ML_sample_geometries[:,0],ML_sample_geometries[:,1],color="magenta",marker="*")

im2=grid[1,0].pcolormesh(x, y, E_10_err, cmap=cmap,shading='auto',vmin=z_min,vmax=np.amax(E_10_err),alpha=alpha)
grid[1,0].set_title("truncated sum (10%)")
grid[1,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[1,0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[1,0].scatter(ML_sample_geometries[:,0],ML_sample_geometries[:,1],color="magenta",marker="*")

im3=grid[1,1].pcolormesh(x, y, E_20_err, cmap=cmap,shading='auto',vmin=z_min,vmax=np.amax(E_20_err),alpha=alpha)
grid[1,1].set_title("truncated sum (20%)")
grid[1,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[1,1].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[1,1].scatter(ML_sample_geometries[:,0],ML_sample_geometries[:,1],color="magenta",marker="*")








im1=grid[0,1].pcolormesh(x, y, E_ML_auto_err, cmap=cmap,shading='auto',vmin=z_min,vmax=np.amax(E_ML_auto_err),alpha=alpha)

grid[0,1].scatter(ML_top_sample_geometries[:,0],ML_top_sample_geometries[:,1],color="magenta",marker="*")

grid[0,1].set_title("GP (auto)")
grid[0,1].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
plt.suptitle("Absolute deviation from CCSD energy")
colorbar=fig.colorbar(im1,label='Error (mHartree)')
colorbar=fig.colorbar(im0,label='Error (mHartree)')
colorbar=fig.colorbar(im2,label='Error (mHartree)')
colorbar=fig.colorbar(im3,label='Error (mHartree)')

plt.tight_layout()
plt.savefig("plots/BeH2_energies.pdf")
plt.show()

fig,grid=plt.subplots(2,2,sharey=True,sharex=True,figsize=(15,10))
im0=grid[0,0].pcolormesh(x, y, niter_ML-niter_MP2, cmap=cmap,shading='auto',vmin=np.amin(niter_ML-niter_MP2),vmax=np.amax(niter_ML-niter_MP2),alpha=alpha)
grid[0,0].set_title("GP")
grid[0,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[0,0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0,0].scatter(ML_sample_geometries[:,0],ML_sample_geometries[:,1],color="magenta",marker="*")

im2=grid[1,0].pcolormesh(x, y, niter_AMP_10-niter_MP2, cmap=cmap,shading='auto',vmin=np.amin(niter_AMP_10-niter_MP2),vmax=np.amax(niter_AMP_10-niter_MP2),alpha=alpha)
grid[1,0].set_title("truncated sum (10%)")
grid[1,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[1,0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[1,0].scatter(ML_sample_geometries[:,0],ML_sample_geometries[:,1],color="magenta",marker="*")

im3=grid[1,1].pcolormesh(x, y, niter_AMP_20-niter_MP2, cmap=cmap,shading='auto',vmin=np.amin(niter_AMP_20-niter_MP2),vmax=np.amax(niter_AMP_20-niter_MP2),alpha=alpha)
grid[1,1].set_title("truncated sum (20%)")
grid[1,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[1,1].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[1,1].scatter(ML_sample_geometries[:,0],ML_sample_geometries[:,1],color="magenta",marker="*")


im1=grid[0,1].pcolormesh(x, y, niter_ML_auto-niter_MP2, cmap=cmap,shading='auto',vmin=np.amin(niter_ML_auto-niter_MP2),vmax=np.amax(niter_ML_auto-niter_MP2),alpha=alpha)

grid[0,1].scatter(ML_top_sample_geometries[:,0],ML_top_sample_geometries[:,1],color="magenta",marker="*")

grid[0,1].set_title("GP (auto)")
grid[0,1].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
plt.suptitle("Absolute deviation from CCSD energy")
colorbar=fig.colorbar(im1,label=r'$\Delta$ num. iter.')
colorbar=fig.colorbar(im0,label=r'$\Delta$ num. iter.')
colorbar=fig.colorbar(im2,label=r'$\Delta$ num. iter.')
colorbar=fig.colorbar(im3,label=r'$\Delta$ num. iter.')

plt.tight_layout()
plt.savefig("plots/BeH2_niter.pdf")
niter_ML_auto=np.array(niter_ML_auto,dtype=float)
niter_ML=np.array(niter_ML,dtype=float)
niter_AMP_10=np.array(niter_AMP_10,dtype=float)
niter_AMP_20=np.array(niter_AMP_20,dtype=float)
niter_ML_auto[niter_ML_auto == 1] = np.nan
niter_AMP_10[niter_AMP_10 == 1] = np.nan
niter_ML[niter_ML == 1] = np.nan
niter_AMP_20[niter_AMP_20 == 1] = np.nan
print("Average number of iterations")
print("MP2: %f"%np.nanmean(niter_MP2))
print("ML auto: %f"%np.nanmean(niter_ML_auto))
print("ML: %f"%np.nanmean(niter_ML))
print("AMP 20: %f"%np.nanmean(niter_AMP_20))
print("AMP 10: %f"%np.nanmean(niter_AMP_10))
plt.show()
