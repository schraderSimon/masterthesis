import sys
sys.path.append("../libraries")
from func_lib import *
molecule_name="BeH2_asymmetric"
basis="cc-pVDZ"
len_sample_geom=25
type="U"
#type="avstand"
file="energy_data/convergence_%s2D_%s_%d_%s.bin"%(molecule_name,basis,len_sample_geom,type)
import pickle
with open(file,"rb") as f:
    energy_dict=pickle.load(f)
print(energy_dict["std"].shape)
std_average=np.mean(energy_dict["std"],axis=0)
print(std_average.shape)
std_average=std_average.reshape((10,10))
CCSD=np.array(energy_dict["E_CCSD"]).reshape((10,10))

file2="energy_data/Coulomb_test.bin"
import pickle
with open(file2,"rb") as f:
    energy_dict2=pickle.load(f)
AMPred=np.array(energy_dict2["E_machineLearn"]).reshape(10,10)
print(AMPred)
sys.exit(1)
E_ML=np.array(energy_dict["E_machineLearn"]).reshape(10,10)
#x=energy_dict["x"]
#y=energy_dict["y"]
x=y=np.linspace(2,6,10)
sample_geom=energy_dict["sample_geometries"]
test_geom=energy_dict["test_geometries"]
AMPerr=np.abs(AMPred-CCSD)*1000
MLerr=np.abs(E_ML-CCSD)*1000
print(E_ML-CCSD)
cmap="OrRd"
cmap="RdPu"
cmap="jet"
#cmap="gnuplot"
alpha=1
z_min=0
z_max=np.amax( np.concatenate( (AMPerr.ravel(),MLerr.ravel(),[1.6]) ) )
fig,grid=plt.subplots(1,2,sharey=True,sharex=True,figsize=(20,6))
im0=grid[0].pcolormesh(x, y, AMPerr, cmap=cmap,shading='auto',vmin=z_min,vmax=z_max,alpha=alpha)
grid[0].set_title("Parameter learning (50%)")
#grid[0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0].scatter(sample_geom[:,0],sample_geom[:,1],s=60,color="white",marker="*")

im1=grid[1].pcolormesh(x, y, MLerr, cmap=cmap,shading='auto',vmin=z_min,vmax=z_max,alpha=alpha)
grid[1].set_title("Machine Learning")
grid[1].scatter(sample_geom[:,0],sample_geom[:,1],s=60,color="white",marker="*")
grid[1].set_xlabel(r"distance $H^2$-Be (Bohr)")
for i in range(2):
        grid[i].plot(2,2,"o",color="magenta",label="Ref. geom.")
plt.tight_layout()
fig.subplots_adjust(right=0.8)
colorbar=fig.colorbar(im1,label='Error (mHartree)')

niter_CCSD=np.array(energy_dict["MP2"],dtype="float")
niter_AMP_startguess=np.array(energy_dict["EVC"],dtype="float")
niter_machinelearn_guess=np.array(energy_dict["GP"],dtype="float")
plt.savefig("BeH2_stretch_convergence.pdf")
plt.show()

fig,grid=plt.subplots(1,2,sharey=True,sharex=True,figsize=(15,6))


#niter_CCSD[[0,9,90,99]]=niter_AMP_startguess[[0,9,90,99]]=niter_machinelearn_guess[[0,9,90,99]]=np.nan
niter_CCSD=niter_CCSD.reshape((10,10))
niter_AMP_startguess=niter_AMP_startguess.reshape((10,10))-niter_CCSD
niter_machinelearn_guess=niter_machinelearn_guess.reshape((10,10))-niter_CCSD
im0=grid[0].pcolormesh(x, y, niter_AMP_startguess, cmap=cmap,shading='auto',alpha=alpha)
grid[0].set_title("Parameter learning (50%)")
#grid[0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0].scatter(sample_geom[:,0],sample_geom[:,1],s=60,color="white",marker="*")

im1=grid[1].pcolormesh(x, y, niter_machinelearn_guess, cmap=cmap,shading='auto',alpha=alpha)
grid[1].set_title("Machine Learning")
grid[1].scatter(sample_geom[:,0],sample_geom[:,1],s=60,color="white",marker="*")
grid[1].set_xlabel(r"distance $H^2$-Be (Bohr)")
colorbar=fig.colorbar(im1,label='Error (mHartree)')

plt.show()

fig,grid=plt.subplots(1,1,sharey=True,sharex=True,figsize=(15,6))
im1=grid.pcolormesh(x, y, std_average, cmap=cmap,shading='auto',alpha=alpha)
colorbar=fig.colorbar(im1,label='Error (mHartree)')
plt.show()
