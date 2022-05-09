import sys
sys.path.append("../libraries")
#from quantum_library import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from func_lib import *
import pickle
def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
basis="STO-6G"

file1="energy_data/BeH2_stretch_sample_-2.bin"

with open(file1,"rb") as f:
    data=pickle.load(f)
file2="energy_data/BeH2_stretch_sample_-5.bin"
with open(file2,"rb") as f:
    data2=pickle.load(f)

x_of_interest=data["xvals"]
EVC_approx=[data["EVC_approx"],data2["EVC_approx"]]
EVC_std=[data["EVC_approx_std"],data2["EVC_approx_std"]]
num_measurements=[data["num_measurements"],data2["num_measurements"]]
Es=[data["Es"],data2["Es"]]
stds=[data["stds"],data2["stds"]]
E_EVC=np.array([-15.330116689766427, -15.444712546470777, -15.533961909877856, -15.602722991823345, -15.654900971190193, -15.693634139607733, -15.721448791887939, -15.740384792163326, -15.752095769594687, -15.757928437889289, -15.758985173340202, -15.756173336311726, -15.750244161865599, -15.741823469751141, -15.731435967851118, -15.71952453448766, -15.70646555292266, -15.692581126173842, -15.678148813162451, -15.663409390072207, -15.648573045629407, -15.633824370558145, -15.619326382251684, -15.605224111904077, -15.591647457421324, -15.578714386813553, -15.5665336959343, -15.555207645877065, -15.54483357492565, -15.535503045247301, -15.527296392095082, -15.520270930977162, -15.514443919509155, -15.50977659472081, -15.50616912806937, -15.503472433840583, -15.501512658298974, -15.500117363014574, -15.499134477022341, -15.498441615489194, -15.497947916959038, -15.497591527298107, -15.497334940866399, -15.49715926609256, -15.497057776913277, -15.497028916847672, -15.497069145072825, -15.497166507647565, -15.497296318912525, -15.497420406681387, -15.49749064683723])
chem_acc=1.6*1e-3
fig, ax= plt.subplots(1, 2,figsize=(10,5),sharey=True)
from matplotlib import ticker
ax[0].set_yticks(np.arange(-15.335, -15.325, step=0.002))
formatter = ticker.FormatStrFormatter('%.4f')
ax[0].yaxis.set_major_formatter(formatter)
ax[0].set_ylabel("Energy (Hartree)")
plt.suptitle("Convergence ")
epsilons=["10^{-2}","10^{-5}"]
locs=["left","right"]
for i in range(2):
    ax[i].set_title(r"$\epsilon=%s$"%epsilons[i],loc=locs[i])
    ax[i].axhline(E_EVC[0],label=r"$E_{exact}$",color="green")
    ax[i].fill_between([-1,1e16],E_EVC[0]-chem_acc,E_EVC[0]+chem_acc,color="green",alpha=0.2,label="Chemical\n accuracy")
    ax[i].set_ylim([-15.335,-15.325])
    ax[i].set_xlim([num_measurements[i][0],num_measurements[i][-1]])
    #ax[i].set_yticks(np.arange(-15.335, -15.325, step=0.001))
    formatter = ticker.FormatStrFormatter('%.4f')
    ax[i].yaxis.set_major_formatter(formatter)
    ax[i].plot(num_measurements[i],Es[i],label=r"$\bar {E}_0$",color="r")
    ax[i].fill_between(num_measurements[i], Es[i] - stds[i], Es[i] + stds[i],
                     color='r', alpha=0.2,label=r"$\pm\sigma_{E_0}$")
    ax[i].set_xlabel(r"$N_{measurements}$")
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.55,0.735),loc="center",handletextpad=0.1,labelspacing = 0.0)

plt.tight_layout()
plt.savefig("resultsandplots/EVC_convergence_epsilon.pdf")
plt.show()
fig, ax= plt.subplots(1, 2,figsize=(10,5),sharey=True)
#from matplotlib import ticker
#ax[0].set_yticks(np.arange(-15.335, -15.325, step=0.002))
#formatter = ticker.FormatStrFormatter('%.4f')
#ax[0].yaxis.set_major_formatter(formatter)
ax[0].set_ylabel("Energy (Hartree)")
plt.suptitle("Error along PES")
epsilons=["10^{-2}","10^{-5}"]
locs=["left","right"]

for i in range(2):
    ax[i].axhline(0,color="green")
    ax[i].set_title(r"$\epsilon=%s$"%epsilons[i],loc=locs[i])
    error=np.array(EVC_approx[i]-E_EVC)
    ax[i].plot(x_of_interest,error,label=r"$\bar {E}_0-E_{exact}$",color="r")
    #ax[i].fill_between(x_of_interest, error -EVC_std[i], error +EVC_std[i],color='r', alpha=0.2,label=r"$\pm\sigma_{E_0}$")
    ax[i].set_xlabel("x (Bohr)")

    ax[i].fill_between(x_of_interest,-chem_acc,chem_acc,color="green",alpha=0.2,label="Chemical\n accuracy")
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.55,0.735),loc="center",handletextpad=0.1,labelspacing = 0.0)

plt.tight_layout()
plt.savefig("resultsandplots/EVC_energy_epsilon.pdf")
plt.show()
