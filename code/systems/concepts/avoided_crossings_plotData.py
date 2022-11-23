import sys
sys.path.append("../libraries")
from func_lib import *
import pickle
import matplotlib as mpl
file="orbitals_data/avoided_crossings.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
noons_data=data["noons"]
energies_data=data["energies"]
cl=crossing_locations=data["crossings"]
#cl=crossing_locations=[[[1.95,2.45],[4.32,0.66]],[[3.095,0.000453],[4.32,0.000359]]]

acl=avoided_crossing_locations=data["avoided_crossings"]
acl=avoided_crossing_locations=[[[1.67,1.5],[2.85,0.99]],[[3.35,0.0044],[3.95,6*1e-4]]]
cl=crossing_locations=[[[1.95,2.45],[4.32,0.66]],[[3.095,0.000453],[4.32,0.000359]]]

xs=data["xs"]
fig,axes=plt.subplots(2,1,sharey=False,figsize=(7,10))
axes[0].set_title("Orbital energy (Hartree)")
axes[1].set_title("Natural occupation number")
color_samplings=np.linspace(0.0,0.8,len(noons_data[0,:]))
#color_samplings=np.linspace(0.3,1,len(noons_data[0,:]))

cmap=mpl.colormaps["viridis"]
np.random.seed(0)
cmap_values=cmap(color_samplings)
np.random.shuffle(cmap_values)


for i in range(len(noons_data[0,:])):
    if np.array(noons_data)[:,i][-1]>3.1*1e-4:
        axes[1].plot(xs,np.array(noons_data)[:,i],color=cmap_values[i],linewidth=3)#[:,5:])
    if 0.651<np.array(energies_data)[:,i][-1]<0.652:
        axes[0].plot(xs,np.array(energies_data)[:,i],color=cmap(color_samplings)[-1],linewidth=3)
    else:
        axes[0].plot(xs,np.array(energies_data)[:,i],color=cmap_values[i],linewidth=3)#[:,5:])

axes[0].set_xlabel("x (Bohr)")
axes[1].set_xlabel("x (Bohr)")
axes[0].set_xticks([2,3,4,5])
axes[1].set_xticks([2,3,4,5])
x=xs[-1]
for i in range(2):
    for j in range(2):
        axes[i].plot(acl[i][j][0],acl[i][j][1],marker='$\u25CC$',markerfacecolor ='red',linewidth=1,markeredgewidth=1,markeredgecolor="red",alpha=1.0,ms=x*5) #Avoided crossing
        axes[i].plot(cl[i][j][0],cl[i][j][1],marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="fuchsia",alpha=1.0,ms=x*5) # crossing

axes[1].set_ylim([2.7*1e-4,0.013])

axes[0].set_ylim([-0.25,3])
axes[1].set_yscale("log")
plt.tight_layout()
plt.savefig("avoided_crossings.pdf")
plt.show()
