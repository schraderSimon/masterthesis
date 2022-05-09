import sys
sys.path.append("../libraries")
from pyscf.mcscf import CASSCF
from quantum_library import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def molecule(x):
    return "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
basis="STO-6G"
from func_lib import *
file="energy_data/UCC2_BeH2_stretch.bin"
import pickle
with open(file,"rb") as f:
    dicty=pickle.load(f)
Hs_UCC2=dicty["H"]
Ss_UCC2=dicty["S"]
x=xvals=dicty["xvals"]
sample_x=np.array(dicty["sample_x"])
sample_E_UCC2=np.array(dicty["sample_E"])
E_FCI=np.array(dicty["E_FCI"])
file="energy_data/UCC1_BeH2_stretch.bin"
import pickle
with open(file,"rb") as f:
    dicty=pickle.load(f)
Hs_UCC1=dicty["H"]
Ss_UCC1=dicty["S"]
sample_E_UCC1=np.array(dicty["sample_E"])
print(sample_E_UCC1)
vals=list(np.arange(len(sample_x)))
vals=[0,1,2,3,4,5]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("x (Bohr)")
axes[1][1].set_xlabel("x (Bohr)")
CCEVC_energies=[[],[],[]]
print(sample_x)
UCC_1_energies=[-15.486052941912824, -15.484700700420767, -15.483367514366726, -15.48210185770184, -15.480952343345953, -15.479965236096927, -15.47918388980364, -15.47864906550786, -15.47840006386915, -15.478476563919205, -15.478920987812852, -15.479781101551861, -15.4811125168981, -15.482980768543722, -15.485462615757974, -15.488646100499633, -15.492628700999017, -15.497512711822164, -15.50339690198127, -15.510363898850908, -15.518464383061465, -15.527702744993903, -15.538032381770655, -15.54936589460936, -15.561594392051276, -15.574603495235266, -15.58827975680116, -15.602509705717178, -15.617175826106465, -15.632152006659854, -15.647299310485385, -15.66246193184966, -15.67746311961357, -15.692100658927549, -15.706141583145119, -15.719315768015722, -15.731308039581263, -15.741748370551585, -15.750199648843285, -15.756142363162489, -15.758955363846262, -15.757891609150532, -15.75204749257788, -15.740323949392797, -15.721377057324503, -15.693555269764596, -15.654819794941357, -15.602644114456602, -15.533888667291883, -15.444648121749545, -15.3300720751391][::-1]
UCC_2_energies=[-15.497492144321853, -15.49745101985824, -15.497410718536207, -15.497373870188401, -15.497344954273448, -15.49733044885481, -15.49733967621832, -15.497385973099508, -15.497488158837756, -15.49767308323392, -15.497976410365263, -15.498448285407314, -15.499155682354479, -15.500188057844882, -15.50165538573848, -15.503692897570554, -15.50645214774505, -15.510090563567674, -15.514745246360455, -15.52052250191721, -15.527473299838148, -15.535605036309468, -15.544877586624557, -15.555223429755195, -15.566554590831363, -15.578770478000356, -15.591761420220289, -15.605409479730017, -15.61958771643796, -15.634158447384188, -15.648970859550934, -15.66385797769995, -15.678633087162817, -15.693085372818189, -15.706974588337754, -15.720024578113, -15.731915191861933, -15.742272370585548, -15.750655720251656, -15.756543142565729, -15.759311367357945, -15.758211773836415, -15.75233945345314, -15.740594468571626, -15.721632176333488, -15.693800705838967, -15.655060986918304, -15.60288680839194, -15.534138963694122, -15.444912717164147, -15.330359277693688][::-1]

sample_points=[[[0,10,11,13],[1,2,3,4,5,6,7,8,9,10,11,13]],[[0,6,10,11,12,13],[1,2,3,4,5,6,7,8,9,13]]]
sample_points=[[[0,10,11,12,13],[0,10,11,12,13]],[[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],[[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]]]
thresholds=[[1e-3,1e-3],[1e-3,1e-3],[0,0]]
for i in range(3):
    for j in range(2):
        vals=sample_points[i][j]
        E=[]
        for k in range(len(xvals)):
            if j==0:
                H=Hs_UCC1[k][np.ix_(vals,vals)].copy()
                S=Ss_UCC1[k][np.ix_(vals,vals)].copy()
            else:
                H=Hs_UCC2[k][np.ix_(vals,vals)].copy()
                S=Ss_UCC2[k][np.ix_(vals,vals)].copy()

            e,c=canonical_orthonormalization(H,S,threshold=thresholds[i][j]) #lazy way to solve generalized eigenvalue problem
            E.append(e)
        print("%d %d "%(i,j))
        for eigval in np.linalg.eigh(S)[0]:
            print(eigval)
        CCEVC_energies[i].append(E)
for i in range(2):
    for j in range(2):
        axes[i][j].axvline(x=2,linestyle="--",color="gray",label="Ref. geom.",linewidth=2)
        axes[j][i].plot(x,E_FCI,label="FCI",color="tab:purple")
        if j==0:
            axes[j][i].plot(x,UCC_1_energies,label="1-UCC",color="tab:green",alpha=0.7)
            axes[j][i].plot(x,CCEVC_energies[i][j],"--",label=r"EVC, $\epsilon=10^{-3}$",color="tab:orange")
            if i==1:
                axes[j][i].plot(x,CCEVC_energies[2][j],"--",label=r"EVC, $\epsilon=0$",color="tab:red")
            axes[j][i].plot(sample_x[sample_points[i][j]],sample_E_UCC1[sample_points[i][j]],"*",label="Smp. pts.",color="black",markersize=9)

        if j==1:
            axes[j][i].plot(x,UCC_2_energies,label="2-UCC",color="tab:blue",alpha=0.7)
            axes[j][i].plot(x,CCEVC_energies[i][j],"--",label=r"EVC, $\epsilon=10^{-3}$",color="tab:orange")
            if i==1:
                axes[j][i].plot(x,CCEVC_energies[2][j],"--",label=r"EVC, $\epsilon=0$",color="tab:red")
            axes[j][i].plot(sample_x[sample_points[i][j]],sample_E_UCC2[sample_points[i][j]],"*",label="Smp. pts.",color="black",markersize=9)

        axes[j][i].grid()
        axes[j][i].set_ylim([-15.77,-15.32])
handles, labels = axes[-1][-1].get_legend_handles_labels()
handles2, labels2 = axes[0][0].get_legend_handles_labels()
handles.append(handles2[1])
labels.append(labels2[1])
axes[0][1].legend(bbox_to_anchor=(1.72,0.58),loc="center right",handletextpad=0.1,labelspacing = 0.0)
axes[1][1].legend(bbox_to_anchor=(1.72,0.58),loc="center right",handletextpad=0.1,labelspacing = 0.0)

#fig.legend(handles, labels, bbox_to_anchor=(0.45,0.62),loc="center",handletextpad=0.3,labelspacing = 0.1)

plt.tight_layout()
fig.subplots_adjust(right=0.78)

plt.savefig("resultsandplots/BeH2_stretch_QC.pdf")
plt.show()
print(CCEVC_energies[0][1])
