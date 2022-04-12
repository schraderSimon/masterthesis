import pickle
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *

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

fig,axes=plt.subplots(2,2,sharey=False,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("distance (Bohr)")
axes[1][1].set_xlabel("distance (Bohr)")
#axes[0][0].set_ylim([-100.2,-99.95])
axes[0][0].set_title("6-31G*")
axes[0][0].plot(x,RHF,"--",label="RHF",color="tab:cyan")
axes[0][0].plot(x,CCSD,label="CCSD",color="tab:purple")
for k,i in enumerate(range(13,1,-3)):
    axes[0][0].plot(x,energies_EVC[k],"--",label="EVC (%d)"%i)

axes[0][0].grid()


file="HF_data_cc-pVTZ.bin"
with open(file,"rb") as f:
    data=pickle.load(f)

x=data["x"]
sample_strengths=data["strenghts"]
RHF=data["RHF"]
CCSD=data["CC"]
energies_EVC=[]
for i in range(13,1,-3):
    energies_EVC.append(data["%d"%i])

axes[0][1].set_title("cc-pVTZ")
axes[0][1].plot(x,RHF,"--",label="RHF",color="tab:cyan")
axes[0][1].plot(x,CCSD,label="CCSD",color="tab:purple")
for k,i in enumerate(range(13,1,-3)):
    axes[0][1].plot(x,energies_EVC[k],"--",label="EVC (%d)"%i)

axes[0][1].grid()

handles, labels = axes[0][0].get_legend_handles_labels()
plt.legend()
plt.tight_layout()
plt.show()
