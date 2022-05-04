import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
file="energy_data/HF_Natorb_0.50.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
x=data["xval"]
E_CCSD=data["CCSD"]
sample_geometry=data["samples"]
print(sample_geometry)
sample_energies=data["energy_samples"]
E_WF=data["WF"]
E_AMP_full=data["AMP"]
E_AMP_red=data["AMPred"]
titles=data["titles"]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
#axes[0][0].set_yticks(np.linspace(-99.95,-100.2,6))
#axes[1][0].set_yticks(np.linspace(-99.95,-100.2,6))
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("x (Bohr)")
axes[1][1].set_xlabel("x (Bohr)")
axes[0][0].set_ylim([-100.35,-100.0])
axes[1][1].axvline(x=1.75,linestyle="--",color="gray",label="Ref. geom.",linewidth=2)

for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):

        axes[i][j].plot(x,E_CCSD[i][j],alpha=0.5,label="CCSD",color="tab:cyan")
        axes[i][j].plot(x,E_AMP_full[i][j],"--",label="AMP-CCEVC",color="tab:red")
        axes[i][j].plot(x,E_AMP_red[i][j],"--",label=r"AMP, $(p_v=50\%)$",color="tab:green")
        axes[i][j].plot(x,E_WF[i][j],"--",label="WF-CCEVC",color="tab:orange")
        print(i,j)
        axes[i][j].plot(sample_geometry[i][j],sample_energies[i][j],"*",color="black",label="Sample points",markersize=9)
        axes[i][j].set_title(titles[i][j])
        axes[i][j].grid()
handles, labels = axes[-1][-1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.0,0.51),loc="lower right",handletextpad=0.3,labelspacing = 0.1)
fig.tight_layout()
fig.subplots_adjust(right=0.90)
plt.savefig("resultsandplots/HF_natorb.pdf")
plt.show()
