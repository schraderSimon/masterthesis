import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from matrix_operations import *
from helper_functions import *
molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
basis = 'cc-pVDZ'

file="../coupled_cluster/energy_data/BeH2_CCSD_rawdata.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
print("FCI")
print(list(data["FCI"]))
FCI=data["FCI"]
xvals=np.linspace(0,4,81)
t1s_1,t2s_1,l1s_1,l2s_1,sample_energies_1=data["CC_1"]
CC_left=list(sample_energies_1)
t1s_2,t2s_2,l1s_2,l2s_2,sample_energies_2=data["CC_2"]
CC_right=list(sample_energies_2[::-1])
file="../groundstate_HF_pos/energy_data/BeH2_data.bin"
with open(file,"rb") as f:
    data=pickle.load(f)

x=data["xc_array"]
phi1=data["Phi1_E"]
phi2=data["Phi2_E"]
fig,axes=plt.subplots(1,2,sharey=False,sharex=True,figsize=(12,6))
axes[0].plot(x,phi1,label=r"RHF $|\Phi_1 \rangle$",color="Tab:blue")
axes[0].plot(x,phi2,label=r"RHF $|\Phi_2 \rangle$",color="Tab:cyan")
axes[0].plot(xvals,FCI,label=r"FCI",color="Tab:purple",alpha=0.8)
axes[0].plot(xvals,CC_left,"--",label=r"$e^{T} |\Phi_1 \rangle$",color="Tab:orange",alpha=0.8)

axes[0].plot(xvals,CC_right,"--",label=r"$e^{T} |\Phi_2 \rangle$",color="Tab:red",alpha=0.8)
axes[0].set_ylim([-15.85,-15.50])
axes[0].set_xlim([-0.1,4.1])

axes[0].set_xlabel("x (Bohr)")
axes[0].set_ylabel("E (Hartree)")
axes[0].set_title("Absolute energy")
axes[1].plot(xvals,1000*(CC_left-FCI),label=r"$e^{T} |\Phi_1 \rangle$",color="Tab:orange",alpha=0.8)
axes[1].plot(xvals,1000*(CC_right-FCI),label=r"$e^{T} |\Phi_2 \rangle$",color="Tab:red",alpha=0.8)
axes[1].set_ylabel(r"$\Delta$E (mHartree)")
axes[1].set_xlabel("x (Bohr)")
axes[1].axhline(0,color="black")
axes[1].fill_between([-1,5],1.6,-1.6,color="green",alpha=0.5,label="Chemical\n accuracy")
axes[1].set_title("Error compared to FCI")
#handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend( bbox_to_anchor=(-0.02,1.02),loc="upper left",handletextpad=0.3,labelspacing = 0.1)
axes[1].legend( bbox_to_anchor=(-0.02,1.02),loc="upper left",handletextpad=0.3,labelspacing = 0.1)
plt.suptitle("BeH2 insertion")
fig.tight_layout()
plt.savefig("BeH2_insertion_CC.pdf")
plt.show()
