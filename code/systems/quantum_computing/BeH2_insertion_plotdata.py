import sys
sys.path.append("../libraries")
from pyscf.mcscf import CASSCF
from quantum_library import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
basis="STO-6G"
from func_lib import *
file="energy_data/FCI_BeH2_insertion_HSdata.bin"
import pickle
with open(file,"rb") as f:
    dicty=pickle.load(f)
Hs=dicty["H"]
print(Hs[0][0,0])
Ss=dicty["S"]
x=xvals=dicty["x_of_interest"]
sample_x=dicty["sample_x"]

"""
sample_coeffmatrices_1=[]
sample_coeffmatrices_2=[]
xc_array=xvals
occdict2={"A1":6,"B1":0,"B2":0}
occdict3={"A1":4,"B1":2,"B2":0}
occdict1={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]
energies=np.zeros((len(xc_array),3))
E_FCI=np.zeros(len(xc_array))
correct_ocdicts_sample1=[]
correct_ocdicts_sample2=[]
for k,x in enumerate(xc_array):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    energies_temp=np.zeros(3)
    for i in [0,1,2]:
        mf = scf.RHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        mo_coeff_temp.append(mf.mo_coeff)
        mo_en_temp.append(mf.mo_energy)
        energies_temp[i]=e
    es=np.argsort(energies_temp)
    if basis=="cc-pVDZ" or basis=="STO-3G":
        pass
    else:
        mycas = mcscf.CASSCF(mf, 6, 4)
        e=mycas.kernel()
        print(e[0])
        E_FCI[k]=e[0]
        print("Pos: %f, EFCI: %f"%(x,e[0]))
        pass
    if x<2.5:
        for i in range(3):
            energies[k,i]=energies_temp[es[i]]
        sample_coeffmatrices_1.append(mo_coeff_temp[es[0]])
        sample_coeffmatrices_2.append(mo_coeff_temp[es[1]])
        correct_ocdicts_sample1.append(occdicts[es[0]])
        correct_ocdicts_sample2.append(occdicts[es[1]])
    else:
        for i in range(3):
            energies[k,i]=energies_temp[i]
        sample_coeffmatrices_1.append(mo_coeff_temp[0])
        sample_coeffmatrices_2.append(mo_coeff_temp[1])
        correct_ocdicts_sample1.append(occdicts[0])
        correct_ocdicts_sample2.append(occdicts[1])
print("E_FCI")
print(E_FCI)
sample_geom1=xc_array
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
geom_alphas1=xc_array
geom_alphas=[[x] for x in geom_alphas1]

for i in range(len(sample_coeffmatrices_1)):
    new1=localize_procrustes(mol,sample_coeffmatrices_1[i],mf.mo_occ,ref_mo_coeff=sample_coeffmatrices_1[0],mix_states=False)
    sample_coeffmatrices_1[i]=new1
    new2=localize_procrustes(mol,sample_coeffmatrices_2[i],mf.mo_occ,ref_mo_coeff=sample_coeffmatrices_2[-1],mix_states=False)
    sample_coeffmatrices_2[i]=new2

energies=[]
energies1=[]
for i in range(len(xc_array)):
    print(i)
    atom=molecule(xc_array[i])
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mol.build()
    mf=scf.RHF(mol)
    mf.irrep_nelec=correct_ocdicts_sample1[i]
    e=mf.kernel(verbose=0)

    R=np.linalg.inv(mf.mo_coeff)@sample_coeffmatrices_1[i]
    R_occ=np.linalg.inv(R[:3,:3])
    R_unocc=np.linalg.inv(R[3:,3:])
    mycc = cc.CCSD(mf,frozen=[0])
    mycc.conv_tol_normt = 1e-7
    mycc.level_shift=.2
    mycc.iterative_damping = 0.8
    if i==0:
        mycc.kernel()

    else:
        mycc.kernel(t1=t1,t2=t2)
    t1 = mycc.t1
    t2 = mycc.t2
    energies1.append(mycc.e_tot)
    print(mycc.e_tot)

for i in range(len(xc_array)):
    print(i)
    atom=molecule(xc_array[::-1][i])
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mol.build()
    mf=scf.RHF(mol)
    mf.irrep_nelec=correct_ocdicts_sample2[::-1][i]
    e=mf.kernel(verbose=0)
    mf.mo_coeff=sample_coeffmatrices_2[::-1][i]
    mycc = cc.CCSD(mf,frozen=[0])
    mycc.conv_tol_normt = 1e-7
    mycc.level_shift=.2
    mycc.iterative_damping = 0.8
    if i==0:
        mycc.kernel()

    else:
        mycc.kernel(t1=t1,t2=t2)
    energies.append(mycc.e_tot)
print("FCI:")
print(list(E_FCI))
print("CC 1")
print(list(energies))
print("CC 2")
print(list(energies1))
"""

E_FCI=[-15.758667827480783, -15.759207336161008, -15.758625430339446, -15.7570274510379, -15.754539609535078, -15.751300813999247, -15.747452434302614, -15.74312675678401, -15.738435416262973, -15.733459191185027, -15.728240516958667, -15.722779743925102, -15.71703564654402, -15.710930099107992, -15.704356298836903, -15.69718953892085, -15.689299354259896, -15.680561869235905, -15.67087133058369, -15.660150084050066, -15.648356673130028, -15.635492393114006, -15.621607747915796, -15.606812438869069, -15.59129761267808, -15.575393019739582, -15.559723849975558, -15.554751195639318, -15.550024300663205, -15.545652946276016, -15.541780638544811, -15.538585089993525, -15.536263644961286, -15.534994382661859, -15.534879387387655, -15.53590383959288, -15.537944963590396, -15.540822442509238, -15.544351039153582, -15.55740665213139, -15.5723362748077, -15.587917327688409, -15.603525036136418, -15.618733378087365, -15.633159543958195, -15.646390038770262, -15.657932064347053, -15.667168852383128, -15.673307734557136]
FCI_func=scipy.interpolate.interp1d(xvals,E_FCI,kind="cubic")
E_FCI=FCI_func(xvals)
sample_energies=FCI_func(sample_x)
Left=[-15.758239226873728, -15.758794863654382, -15.75822592244175, -15.756638440339545, -15.754159146222657, -15.750927503199035, -15.747085351797114, -15.742765444229251, -15.738079749557276, -15.7331093500713, -15.72789685568377, -15.722442698139202, -15.71670561520872, -15.710607345942634, -15.704040858007502, -15.696881179215513, -15.68899748404836, -15.680265456304703, -15.670578736494631, -15.659858654719834, -15.648062008016337, -15.635186799970688, -15.621277105531592, -15.606429996435994, -15.590810651141537, -15.574691595685827, -15.558559229897917, -15.553325053264594, -15.54824856812267, -15.543407430989783, -15.538905522542816, -15.534879657753851, -15.531504084475394, -15.528984282396447, -15.527529553037025, -15.527299100196926, -15.528340784162781, -15.530565621778786, -15.533780926998233, -15.547183339113152, -15.56307075021599, -15.57959901568165, -15.596167339665408, -15.612438511549948, -15.627991842020839, -15.64229577577469, -15.654759369865214, -15.664728286503152, -15.67141919411303]
Right=[-15.672898128137955, -15.666738034499316, -15.657475571519326, -15.64590358121278, -15.632639433103476, -15.618176645292314, -15.602928994700758, -15.587277290758035, -15.571638607480763, -15.556606585036414, -15.543308831145916, -15.53963541803473, -15.53656949612349, -15.534301281513764, -15.533041050756767, -15.53297161183247, -15.534182971942926, -15.536623025780303, -15.540115976530489, -15.544426723533675, -15.54932563019039, -15.554623491884213, -15.560175335033213, -15.577445576383854, -15.594606801547753, -15.611019339239474, -15.626394002469757, -15.64057727055613, -15.3544304680223, -15.316377563853539, -15.298454319102978, -15.276662956979527, -15.262282202204961, -15.244935006833414, -15.227817008071042, -15.209790300252719, -15.196453227937164, -15.186250936954862, -15.17020338423529, -15.159349083461374, -15.14990707181579, -15.144809245332697, -15.144980042353406, -15.13997677630663, -15.144459782415295, -15.150951817949169, -15.157031488570366, -15.163621474932082, -15.170303986973138]
Right=Right[::-1]
for i in range(25):
    Right[i]=np.nan
    Left[i]=np.nan
for i in range(42,49,1):
    print(i)
    Right[i]=np.nan
    Left[i]=np.nan

vals=list(np.arange(len(sample_x)))
vals=[0,1,2,3,4,5]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_ylabel("Energy (Hartree)")
axes[1][0].set_xlabel("x (Bohr)")
axes[1][1].set_xlabel("x (Bohr)")
CCEVC_energies=[[],[]]
sample_points=[[[0,1,6,7,8,9,10,15],[0,1,6,7,8,9,10,15]],[[0,1,2,3,4,5],[0,1,2,3,4,5]]]
thresholds=[[0,1e-3],[0,1e-3]]
for i in range(2):
    for j in range(2):
        print(np.linalg.eigh(Ss[0][np.ix_(vals,vals)].copy())[0])

        vals=sample_points[i][j]
        E=[]
        for k in range(len(xvals)):
            H=Hs[k][np.ix_(vals,vals)].copy()
            S=Ss[k][np.ix_(vals,vals)].copy()
            e,c=canonical_orthonormalization(H,S,threshold=thresholds[i][j]) #lazy way to solve generalized eigenvalue problem
            E.append(e)
        CCEVC_energies[i].append(E)
for i in range(2):
    for j in range(2):
        axins=zoomed_inset_axes(axes[i][j], 4.15, loc="upper left")
        axins.set_xlim(2.6, 3.15)
        axins.set_ylim(-15.55, -15.525)
        axins.plot(x,E_FCI,label="FCI",color="tab:purple")
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        axes[i][j].plot(x,E_FCI,label="FCI",color="tab:purple")
        mark_inset(axes[i][j], axins, loc1=2, loc2=4, fc="none", ec="0.5")
        if j==0:
            axins.plot(x,CCEVC_energies[i][j],"--",label=r"EVC, $\epsilon=0$",color="tab:red")
            axes[i][j].plot(x,CCEVC_energies[i][j],"--",label=r"EVC, $\epsilon=0$",color="tab:red")
        if j==1:
            axins.plot(x,CCEVC_energies[i][j],"--",label=r"EVC, $\epsilon=10^{-3}$",color="tab:orange")
            axes[i][j].plot(x,CCEVC_energies[i][j],"--",label=r"EVC, $\epsilon=10^{-3}$",color="tab:orange")
        axes[i][j].grid()
        axes[i][j].axvline(x=0,linestyle="--",color="gray",label="Ref. geom.",linewidth=2)
        axes[i][j].plot(x,Right,"--",label=r"$e^{T} |\Phi_2 \rangle$",color="Tab:green",alpha=0.75)
        axes[i][j].plot(x,Left,"--",label=r"$e^{T} |\Phi_1 \rangle$",color="Tab:blue",alpha=0.75)
        axes[i][j].set_ylim([-15.77,-15.50])
        axes[i][j].plot(sample_x[sample_points[i][j]],sample_energies[sample_points[i][j]],"*",label="Smp. pts.",color="black",markersize=9)
        axins.plot(x,Right,"--",label=r"$e^{T} |\Phi_2 \rangle$",color="Tab:green",alpha=0.75)
        axins.plot(x,Left,"--",label=r"$e^{T} |\Phi_1 \rangle$",color="Tab:blue",alpha=0.75)
        axins.plot(sample_x[sample_points[i][j]],sample_energies[sample_points[i][j]],"*",label="Smp. pts.",color="black",markersize=9)
handles, labels = axes[0][0].get_legend_handles_labels()
handles2, labels2 = axes[1][1].get_legend_handles_labels()
handles.insert(2,handles2[1])
labels.insert(2,labels2[1])
#axes[0][0].legend(bbox_to_anchor=(0.89,0.585),loc="center",handletextpad=0.1,labelspacing = 0.0)
#axes[1][0].legend(bbox_to_anchor=(0.89,0.585),loc="center",handletextpad=0.1,labelspacing = 0.0)
fig.legend(handles, labels, bbox_to_anchor=(0.89,0.585),loc="center",handletextpad=0.0,labelspacing = -0.1)

plt.tight_layout()
plt.savefig("resultsandplots/BeH2_insertion_QC.pdf")
plt.show()
