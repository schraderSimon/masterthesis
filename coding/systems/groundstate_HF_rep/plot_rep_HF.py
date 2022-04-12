import pickle
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
colors=["tab:green","tab:brown","tab:red","tab:orange"]
alphaval=0.8
file="HF_data_6-31Gstjerne.bin"
with open(file,"rb") as f:
    data=pickle.load(f)

x=data["x"]
sample_strengths=data["strenghts"]
RHF=data["RHF"]
CCSD=data["CC"]
energies_EVC=[]
for i in range(13,1,-3):
    energies_EVC.append(data["%d"%i])

fig,axes=plt.subplots(2,2,sharey=False,sharex=False,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("distance (Bohr)")
axes[1][1].set_xlabel("distance (Bohr)")
#axes[0][0].set_ylim([-100.2,-99.95])
axes[0][0].set_title("6-31G*")
axes[0][0].plot(x,RHF,label="RHF",color="tab:cyan")
for k,i in enumerate(range(4,14,3)):
    axes[0][0].plot(x,energies_EVC[3-k],"--",label="EVC (%d)"%i, color=colors[k], alpha=alphaval)
axes[0][0].plot(x,CCSD,label="CCSD",color="magenta")

axes[0][0].grid()


file="HF_data_cc-pVDZ.bin"
with open(file,"rb") as f:
    data=pickle.load(f)

x=data["x"]
RHF=data["RHF"]
CCSD=data["CC"]
energies_EVC=[]
for i in range(13,1,-3):
    energies_EVC.append(data["%d"%i])

axes[0][1].set_title("cc-pVDZ")
axes[0][1].plot(x,RHF,label="RHF",color="tab:cyan")
axes[0][1].plot(x,CCSD,label="CCSD",color="magenta")
for k,i in enumerate(range(4,14,3)):
    axes[0][1].plot(x,energies_EVC[3-k],"--",label="EVC (%d)"%i, color=colors[k], alpha=alphaval)

axes[0][1].grid()

file="BeH2_data_energy_cc-pVDZ.bin"
with open(file,"rb") as f:
    data=pickle.load(f)

x=data["x"]
energies_EVC=[]
print(data.keys())
for i in range(13,1,-3):
    energies_EVC.append(data["%d"%i])

axes[1][1].set_title("cc-pVDZ")
#axes[1][1].plot(x,RHF,"--",label="RHF",color="tab:cyan")
#axes[1][1].plot(x,CCSD,label="CCSD",color="tab:purple")
for k,i in enumerate(range(4,14,3)):
    axes[1][1].plot(x,energies_EVC[3-k],"--",label="EVC (%d)"%i, color=colors[k], alpha=alphaval)

axes[1][1].grid()

file="BeH2_data_energy_6-31G*.bin"
with open(file,"rb") as f:
    data=pickle.load(f)

x=data["x"]
energies_EVC=[]
for i in range(13,1,-3):
    energies_EVC.append(data["%d"%i])

axes[1][0].set_title("6-31G*")
#axes[1][1].plot(x,RHF,"--",label="RHF",color="tab:cyan")
#axes[1][1].plot(x,CCSD,label="CCSD",color="tab:purple")
for k,i in enumerate(range(4,14,3)):
    axes[1][0].plot(x,energies_EVC[3-k],"--",label="EVC (%d)"%i, color=colors[k], alpha=alphaval)

axes[1][0].grid()

file="../coupled_cluster/energy_data/BeH2_CCSD_rawdata.bin"
with open(file,"rb") as f:
    data=pickle.load(f)
FCI_BeH2_pVDZ=data["FCI"]
axes[1][1].plot(x,FCI_BeH2_pVDZ,label="FCI",color="tab:purple")

file="../groundstate_HF_pos/energy_data/BeH2_data.bin"
with open(file,"rb") as f:
    data=pickle.load(f)
x=data["xc_array"]
phi1=data["Phi1_E"]
phi2=data["Phi2_E"]
axes[1][1].plot(x,phi1,label=r"$|\Phi_1\rangle$",color="tab:cyan")
axes[1][1].plot(x,phi2,label=r"$|\Phi_2\rangle$",color="tab:blue")
axes[1][1].set_ylim([-15.84,-15.53])

file="../groundstate_HF_pos/energy_data/BeH2_data_631G*.bin"
with open(file,"rb") as f:
    data=pickle.load(f)
x=data["xc_array"]
phi1=data["Phi1_E"]
phi2=data["Phi2_E"]
FCI_BeH2_631G=data["FCI"]

axes[1][0].plot(x,FCI_BeH2_631G,label=r"FCI",color="tab:purple")

axes[1][0].plot(x,phi1,label=r"RHF $|\Phi_1\rangle$",color="tab:cyan")
axes[1][0].plot(x,phi2,label=r"RHF $|\Phi_2\rangle$",color="tab:blue")
axes[1][0].set_ylim([-15.821,-15.517])

ha, la = axes[0][0].get_legend_handles_labels()

handles, labels = axes[1][0].get_legend_handles_labels()
handles.append(ha[-1])
labels.append(la[-1])
fig.legend(handles, labels, bbox_to_anchor=(1.0,0.40),loc="lower right",handletextpad=0.3,labelspacing = 0.1)

plt.tight_layout()
fig.subplots_adjust(right=0.89)
plt.savefig("HF_BeH2_couple.pdf")
plt.show()
