import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
file="energy_data/HF_data.bin"
import pickle
with open(file,"rb") as f:
    dicty=pickle.load(f)
energies_sample=dicty["energies_sample"]
titles=dicty["titles"]
EVC_energies=dicty["EVC_energies"]
energiesHF=dicty["HF"]
energiesCC=dicty["CCSD"]
kvals=dicty["kvals"]
x=xc_array=dicty["x"]
sample_geometry=dicty["sample_geometry"]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
#axes[0][0].set_yticks(np.linspace(-99.95,-100.2,6))
#axes[1][0].set_yticks(np.linspace(-99.95,-100.2,6))
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("x (Bohr)")
axes[1][1].set_xlabel("x (Bohr)")
types=["--","--","--","--","."]
colors=["tab:blue","tab:red","tab:green","tab:orange"]
for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):
        axes[i][j].plot(x,energiesHF,label="RHF",color="tab:cyan")
        axes[i][j].plot(x,energiesCC,label="CCSD",color="tab:purple")
        for ki,k in enumerate(kvals):
            axes[i,j].plot(xc_array,EVC_energies[i][j][ki],types[ki],color=colors[ki],label="EVC (%d pt.)"%(k))

        axes[i][j].plot(sample_geometry[i][j],energies_sample[i][j],"*",color="black",label="Sample pts.",markersize=9)

        axes[i][j].set_title(titles[i][j])
        axes[i][j].grid()
handles, labels = axes[0][0].get_legend_handles_labels()
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("x (Bohr)")
axes[1][1].set_xlabel("x (Bohr)")
axes[1][1].set_ylim([-100.4,-99.6])
fig.legend(handles, labels, bbox_to_anchor=(1.0,0.42),loc="lower right",handletextpad=0.3,labelspacing = 0.1)
fig.tight_layout()
fig.subplots_adjust(right=0.85)
plt.savefig("HF_EVC.pdf")
plt.show()
