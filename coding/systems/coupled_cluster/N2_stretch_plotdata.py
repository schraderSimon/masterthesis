import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
file="energy_data/N2_stretch1.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
x=data["xval"]
E_CCSD=data["CCSD"]
sample_geometry=data["samples"]
sample_energies=data["energy_samples"]
E_WF=data["WF"]
E_WF[1][0]=np.loadtxt("energy_data/13points_new.txt")
print(E_WF)
E_AMP_full=data["AMP"]
E_AMP_red=data["AMPred"]
#plt.style.use("bmh")

fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
#axes[0][0].set_yticks(np.linspace(-99.95,-100.2,6))
#axes[1][0].set_yticks(np.linspace(-99.95,-100.2,6))
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("distance (Bohr)")
axes[1][1].set_xlabel("distance (Bohr)")
axes[0][0].set_ylim([-109.3,-108.6])
axes[0][0].set_xticks(np.linspace(2,7,6))
for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):
        axes[i][j].plot(x,E_CCSD,label="CCSD",color="tab:cyan",alpha=0.5)
        axes[i][j].plot(x,E_AMP_full[i][j],"--",label="AMP-CCEVC",color="tab:red")
        axes[i][j].plot(x,E_AMP_red[i][j],"--",label=r"AMP, $(p_v=50\%)$",color="tab:green")
        axes[i][j].plot(x,E_WF[i][j],"--",label="WF-CCEVC",color="tab:orange")
        axes[i][j].plot(sample_geometry[i][j][:14],sample_energies[i][j][:14],"*",color="black",label="Sample points",markersize=7)
        axes[i][j].grid()
handles, labels = axes[-1][-1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.0,0.35),loc="lower right",handletextpad=0.3,labelspacing = 0.1)
plt.suptitle(r"N$_2$ EVC")
fig.tight_layout()
fig.subplots_adjust(right=0.90)
plt.savefig("resultsandplots/N2_ccPVDZ.pdf")
plt.show()
