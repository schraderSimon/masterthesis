import sys
sys.path.append("../libraries")
from func_lib import *
file="orbitals_data/data.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
fig,axes=plt.subplots(2,2,sharey=False,sharex=True,figsize=(12,10))
axes[0,0].set_title(r"$||C(x)-C(x_{{ref}})||$")
axes[0,1].set_title(r"$||T_2(x)-T_2(x_{{ref}})||$")
axes[1,0].set_title(r"$|{\langle \Phi^{{SD}}|\Phi^{{HF}}\rangle}|^2$")
axes[1,1].set_title(r"$E_{CCSD}-E_{CCSD}^{canon.} (Hartree)$")
x_sol=data["xval"]
labels=["Procr.", "Sym. Ort.", "Chol.", "G. Procr.", "G. Sym. Ort.", "Nat. orb."]

norms_coefficientmatrix=data["coefficient_norm"]
norms_T2=data["T2_norm"]
E_CC=data["E_CC"]
overlap_to_HF=data["overlap_to_HF"]
for i in range(2):
    for j in range(2):
        axes[i][j].grid()
types=["-","-.","--","--","--",":"]
for i in range(len(labels)):
    if i==2:
        continue
    axes[0,0].plot(x_sol,norms_coefficientmatrix[i,:],types[i],label=labels[i])
    axes[0,1].plot(x_sol,norms_T2[i,:],types[i],label=labels[i])
    axes[1,0].plot(x_sol,overlap_to_HF[i,:]**2,types[i],label=labels[i])
    axes[1,1].plot(x_sol,E_CC[i,:],types[i],label=labels[i])
handles, labels = axes[0][0].get_legend_handles_labels()
axes[1][0].set_xlabel(r"internuclear distance $x$ (Bohr)")
axes[1][1].set_xlabel(r"internuclear distance $x$ (Bohr)")
fig.legend(handles, labels, bbox_to_anchor=(1.0,0.51),loc="lower right",handletextpad=0.3,labelspacing = 0.1)
fig.tight_layout()
fig.subplots_adjust(right=0.85)
plt.savefig("HF_coefficient_norms.pdf")
plt.show()
