import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
file="energy_data/BeH2_data.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
x=data["xc_array"]
phi1=data["Phi1_E"]
phi2=data["Phi2_E"]
FCI=data["FCI"]
energies_3=data["energy3"]
energies_6=data["energy6"]
geometry_energy_pair=data["samples"]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
#axes[0][0].set_yticks(np.linspace(-99.95,-100.2,6))
#axes[1][0].set_yticks(np.linspace(-99.95,-100.2,6))
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("x (Bohr)")
axes[1][1].set_xlabel("x (Bohr)")
for i in range(2):
    for j in range(2):
        axes[i][j].plot(x,phi1,label=r"RHF $|\Phi_1\rangle$",color="tab:cyan")
        axes[i][j].plot(x,phi2,label=r"RHF $|\Phi_2\rangle$",color="tab:blue")
        axes[i][j].plot(x,energies_3[i][j],"--",label="EVC (3 pt.)",color="tab:red")
        axes[i][j].plot(x,energies_6[i][j],"--",label="EVC (6 pt.)",color="tab:orange")
        axes[i][j].plot(x,FCI,label="FCI",color="tab:purple")
        samplex=[geometry_energy_pair[i][j][k][0] for k in range(len(geometry_energy_pair[i][j]))]
        sampley=[geometry_energy_pair[i][j][k][1] for k in range(len(geometry_energy_pair[i][j]))]
        axes[i][j].plot(samplex,sampley,"*",color="black",markersize=9)
        axes[i][j].grid()
handles, labels = axes[-1][-1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.0,0.35),loc="lower right",handletextpad=0.3,labelspacing = 0.1)
fig.tight_layout()
#fig.subplots_adjust(right=0.85)
plt.savefig("BeH2_EVC.pdf")
plt.show()
