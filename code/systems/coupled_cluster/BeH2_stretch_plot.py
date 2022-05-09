import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
file="energy_data/BeH2_2d_plot_NOGUPTRI.bin"
import pickle
with open(file,"rb") as f:
    energy_dict=pickle.load(f)

CCSD=np.array(energy_dict["CCSD"]).reshape((9,9))
FCI=np.array(energy_dict["FCI"]).reshape(9,9)
AMP=np.array(energy_dict["AMP"]).reshape(9,9)
WF=np.array(energy_dict["WF"]).reshape(9,9)
AMPred=np.array(energy_dict["AMPred"]).reshape(9,9)
x=energy_dict["x"]
y=energy_dict["y"]
sample_geom=energy_dict["samples"]
CCSDerr=np.abs(CCSD-FCI)*1000
WFerr=np.abs(WF-FCI)*1000
AMPerr=np.abs(AMPred-FCI)*1000
AMPranderr=np.abs(AMP-FCI)*1000
cmap="OrRd"
cmap="RdPu"
cmap="jet"
cmap="gnuplot"
alpha=1
z_min=0
z_max=np.amax((CCSDerr,WFerr,AMPerr,AMPranderr))
fig,grid=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
im0=grid[0,0].pcolormesh(x, y, CCSDerr, cmap=cmap,shading='auto',vmin=z_min,vmax=z_max,alpha=alpha)
grid[0,0].set_title("CCSD")
#grid[0,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[0,0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0,0].scatter(sample_geom[:,0],sample_geom[:,1],s=60,color="white",marker="*")

grid[0,1].pcolormesh(x, y, WFerr, cmap=cmap,shading='auto',vmin=z_min,vmax=z_max,alpha=alpha)
im1=grid[0,1].set_title("WF-CCEVC")
grid[0,1].scatter(sample_geom[:,0],sample_geom[:,1],s=60,color="white",marker="*")
#grid[0,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
#grid[0,1].set_ylabel(r"distance $H^1$-Be (Bohr)")

im2=grid[1,0].pcolormesh(x, y, AMPerr, cmap=cmap,shading='auto',vmin=z_min,vmax=z_max,alpha=alpha)
grid[1,0].set_title("AMP-CCEVC")
grid[1,0].scatter(sample_geom[:,0],sample_geom[:,1],s=60,color="white",marker="*")
grid[1,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[1,0].set_ylabel(r"distance $H^1$-Be (Bohr)")

im2=grid[1,1].pcolormesh(x, y, AMPranderr, cmap=cmap,shading='auto',vmin=z_min,vmax=z_max,alpha=alpha)
grid[1,1].set_title(" AMP-CCEVC $(p_v=50\%)$")
grid[1,1].scatter(sample_geom[:,0],sample_geom[:,1],s=60,color="white",marker="*")
grid[1,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
#grid[1,1].set_ylabel(r"distance $H^1$-Be (Bohr)")
for i in range(2):
    for j in range(2):
        grid[i,j].plot(2,2,"o",color="magenta",label="Ref. geom.")
plt.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
colorbar=fig.colorbar(im2, cax=cbar_ax,label='Error (mHartree)')
plt.savefig("BeH2_stretch_Procrustes.pdf")
plt.show()
