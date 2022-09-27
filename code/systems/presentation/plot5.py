import sys
sys.path.append("../libraries")
from func_lib import *
file="../coupled_cluster/energy_data/N2_stretch1.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
x=data["xval"]
E_CCSD=data["CCSD"]
sample_geometry=data["samples"]
sample_energies=data["energy_samples"]
E_WF=data["WF"]
E_WF[1][0]=np.loadtxt("../coupled_cluster/energy_data/13points_new.txt")
print(E_WF)
E_AMP_full=data["AMP"]
E_AMP_red=data["AMPred"]
#plt.style.use("bmh")
ref_geo=2
matplotlib.rcParams.update({'font.size': 13})
matplotlib.rcParams.update({'lines.linewidth': 3})
plt.title(r"Potential Energy for N${}_2$ (cc-pVDZ)")
plt.plot(x,E_CCSD,label="CCSD",color="tab:blue",alpha=1)
plt.plot(x,E_AMP_full[1][0],"--",label="AMP-CCEVC",color="red")
plt.plot(x,E_WF[1][0],"--",label="WF-CCEVC",color="tab:orange")
plt.plot(sample_geometry[1][0][:14],sample_energies[1][0][:14],"*",color="black",label="Sample points",markersize=9)
plt.legend()
plt.ylabel("Energy (Hartree)")
plt.xlabel("Internuc. dist. (Bohr)")

plt.tight_layout()
plt.savefig("plot_5.pdf")
plt.show()
