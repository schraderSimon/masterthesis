import sys
sys.path.append("../libraries")
from func_lib import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

file="energy_data/newBeH2data_1.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
H_Left=data["H"]
S_Left=data["S"]
file="energy_data/newBeH2data_2.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
H_Right=data["H"]
S_Right=data["S"]

x=np.linspace(0,4,81)
Left=[-15.835805169351264, -15.835763638802016, -15.835481360402932, -15.834966038143, -15.834226858852146, -15.833274364431686, -15.832120412741453, -15.830778101206594, -15.82926150730806, -15.827585549127846, -15.825765671327703, -15.823817513142398, -15.821756609106732, -15.819597995520082, -15.817355841255363, -15.815043176617086, -15.812671485378932, -15.810250530385938, -15.807788098455678, -15.805289856136762, -15.802759290058136, -15.80019763496458, -15.797603951346142, -15.79497526714316, -15.792306660052812, -15.789591502441214, -15.786821693869236, -15.783987942505135, -15.78108003833863, -15.778087104877415, -15.774998032299546, -15.771801624820036, -15.768486953325748, -15.765043604502985, -15.761461906261879, -15.757733131971266, -15.753849727568426, -15.74980537640775, -15.745595269478065, -15.741216063377752, -15.736666047609319, -15.731945301059234, -15.727055697229968, -15.72200109101639, -15.716787412642674, -15.711422991152089, -15.705918739514319, -15.70028870250415, -15.694550682744707, -15.688727328945433, -15.682847960225418, -15.67695109856356, -15.671089511813404, -15.665338266139933, -15.65980962105774, -15.65467936285246, -15.650232181266368, -15.646922339593305, -15.645385841472727, -15.646209144563509, -15.649440946216915, -15.654494438953854, -15.66063382481295, -15.667322623265152, -15.674244035431233, -15.681222642941822, -15.688160156622702, -15.694997468544072, -15.701693453616613, -15.70821390765545, -15.71452510490894, -15.720590906694813, -15.72637121082127, -15.731821472051374, -15.736892117741085, -15.741528219175073, -15.745668299009445, -15.749243267986584, -15.752174595416685, -15.754372129962738, -15.755731609608523]
Right=[-15.839832562430079, -15.839716817807396, -15.839346050891113, -15.838723893999545, -15.837856202105531, -15.836751055038503, -15.835418722879316, -15.833871655634871, -15.832124241858791, -15.830192549908245, -15.828093977267597, -15.825846821561978, -15.823469730539948, -15.820981184264024, -15.818398945362013, -15.81573951953476, -15.813017681917298, -15.810246066799502, -15.807434849584544, -15.804591520456757, -15.801720759195705, -15.798824426372978, -15.795901636334154, -15.792948934239702, -15.789960447243615, -15.786928281372393, -15.783842602680677, -15.780692111450717, -15.777464836425077, -15.77414741479163, -15.770726390044013, -15.767188101948934, -15.76351907220316, -15.759706290840853, -15.755737456870316, -15.751601174606423, -15.747287234221059, -15.742786668488737, -15.738091974947771, -15.733197133354844, -15.728097724534306, -15.722791257572728, -15.717276904456492, -15.711556032482271, -15.705632630068344, -15.699513525939075, -15.693209950500497, -15.686739226474478, -15.680128596523035, -15.673421851407381, -15.666692887714325, -15.660071855691763, -15.653799788465806, -15.648320508170228, -15.644379632434427, -15.642852765154426, -15.644059846773159, -15.647444809402204, -15.652192149275221, -15.657708250645737, -15.663644739179782, -15.669804422492758, -15.676071873786931, -15.682374884832857, -15.688664933558691, -15.694906002350958, -15.701068650970326, -15.707126305285335, -15.713053140006904, -15.718822318815896, -15.72440495996496, -15.729769028549528, -15.734878639136243, -15.739693045938504, -15.744165905318589, -15.748244271716615, -15.751867478372397, -15.75496587821485, -15.757459344604248, -15.759255336009238, -15.76024681283893]
Left=np.array(Left)
Right=np.array(Right)

E_FCI=np.array([-15.83643, -15.836382000151804, -15.836093000000002, -15.835571999848204, -15.834827999999998, -15.833870750455393, -15.832713, -15.831368248330216, -15.82985, -15.828172256223748, -15.826351000000003, -15.824402476774804, -15.822342, -15.820184461677046, -15.817944000000002, -15.815633926517016, -15.813265000000001, -15.81084695725489, -15.808388, -15.80589486946342, -15.80337, -15.800814314891426, -15.798227, -15.795605995970874, -15.792946, -15.790240451225076, -15.787481, -15.784658574128823, -15.781762999999998, -15.778783752259619, -15.775709999999998, -15.772530791832697, -15.769235, -15.765811830409596, -15.762252, -15.758547011528918, -15.75469, -15.750674873474729, -15.746497, -15.74215274457216, -15.737641000000002, -15.732961898236637, -15.728118000000002, -15.723113287481308, -15.717955, -15.712652201838138, -15.707218000000001, -15.701668155166148, -15.696024999999999, -15.69031605249726, -15.684583, -15.678880634844802, -15.673301999999998, -15.667978158123514, -15.663154000000004, -15.659191107661151, -15.656804, -15.656621286231896, -15.658574999999999, -15.66234587241126, -15.667306, -15.672839849123065, -15.678689999999996, -15.68470310609647, -15.690784000000003, -15.696859476491067, -15.702886000000001, -15.708828237939276, -15.714654, -15.720331821751822, -15.725829999999998, -15.731115600053432, -15.736151, -15.740895528034445, -15.745301000000001, -15.749315162808788, -15.752877, -15.755918570730408, -15.758353, -15.760088679269597, -15.761034])
fig,axes=plt.subplots(3,2,sharey=True,sharex=True,figsize=(12,15))
axes[0][0].set_ylabel("Energy (Hartree)")
#axes[0][0].set_yticks(np.linspace(-99.95,-100.2,6))
#axes[1][0].set_yticks(np.linspace(-99.95,-100.2,6))
axes[1][0].set_ylabel("Energy (Hartree)")
axes[2][0].set_ylabel("Energy (Hartree)")
axes[2][0].set_xlabel("x (Bohr)")
axes[2][1].set_xlabel("x (Bohr)")
xy=10
lowerleft=[25+xy,26+xy,27+xy,28+xy,29+xy,30+xy,31+xy,32+xy,74,77,76,75,79,78,80]
lowerright=[0,20,21,22,23,24,25,76,77,78,79,80]


sample_points=[[list(np.arange(0,81,10)),list(np.arange(0,81,10))],[[18,19,20,21,22,23,24,25],list(np.arange(65,75,1))],[lowerleft,lowerright]]
CCEVC_energies=[[],[],[]]
sample_energies=[[],[],[]]
exc_lowerright=[]
exc_upperight=[]
for i in range(3):
    for j in range(2):
        if j==0:
            sample_energies[i].append(Left[sample_points[i][j]])
        if j==1:
            sample_energies[i].append(Right[sample_points[i][j]])
for i in range(3):
    for j in range(2):
        vals=sample_points[i][j]
        if j==0:
            Hs=H_Left
            Ss=S_Left
        elif j==1:
            Hs=H_Right
            Ss=S_Right
        E=[]
        for k in range(len(x)):
            H=Hs[k][np.ix_(vals,vals)].copy()
            S=Ss[k][np.ix_(vals,vals)].copy()
            exponent=14.3
            if j==0 and k==0:
                print("i is %d"%i)
                print(S)
                U,s,Vt=scipy.linalg.svd(S)
                #s,c=scipy.linalg.eig(S)
                print(np.sum(s))
                for singval in s:
                    print(singval)
            #eigvals=np.real(scipy.linalg.eig(scipy.linalg.pinv(S,atol=10**(-exponent))@H)[0])
            #eigvals=np.real(scipy.linalg.eig(a=H,b=S+np.eye(len(S))*10**(-exponent))[0])
            try:
                eigvals=np.real(scipy.linalg.eig(a=H,b=S+10**(-exponent)*scipy.linalg.expm(-S/10**(-exponent)))[0])
            except:
                sys.exit(1)
                eigvals=np.real(scipy.linalg.eig(a=H,b=S+np.eye(len(S))*10**(-exponent))[0])
            sorted=np.sort(eigvals)
            if i==2 and j==1:
                exc_lowerright.append(sorted[1])
            if i==0 and j==1:
                exc_upperight.append(sorted[1])
            E.append(sorted[0])
        CCEVC_energies[i].append(E)
for i in range(3):
    for j in range(2):
        if j==0:
            axes[i][j].axvline(x=0,linestyle="--",color="gray",label="Ref. geom.",linewidth=2)
        if j==1:
            axes[i][j].axvline(x=4,linestyle="--",color="gray",label="Ref. geom.",linewidth=2)
        axes[i][j].plot(x,E_FCI,label="FCI",color="tab:purple")
        axes[i][j].plot(x,Right,label=r"$e^{T} |\Phi_2 \rangle$",color="Tab:blue",alpha=0.8)
        axes[i][j].plot(x,Left,label=r"$e^{T} |\Phi_1 \rangle$",color="Tab:green",alpha=0.8)

        if j==1:
            axes[i][j].plot(x,CCEVC_energies[i][j],"--",label=r"EVC $|\Phi_2\rangle$",color="tab:red")
        if j==0:
            axes[i][j].plot(x,CCEVC_energies[i][j],"--",label=r"EVC $|\Phi_1\rangle$",color="tab:orange")
        axes[i][j].set_ylim([-15.84,-15.635])

        axes[i][j].grid()
        if i==2 and j==1:
            pass
            #axes[i][j].plot(x,exc_lowerright,"--",label=r"Exc. $|\Phi_2 \rangle$",color="magenta",alpha=0.75)
        if i==0 and j==1:
            axes[i][j].plot(x,exc_upperight,"--",color="magenta",alpha=0.75)
        axes[i][j].plot(x[sample_points[i][j]],sample_energies[i][j],"*",label="Smp. pts.",color="black",markersize=9)
        if i==2:
            axins=zoomed_inset_axes(axes[i][j], 3, loc="upper left")
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            axins.set_xlim(2.6, 3.15)
            axins.set_ylim(-15.68, -15.64)
            axins.plot(x,E_FCI,label="FCI",color="tab:purple")
            axins.plot(x,Right,label=r"$e^{T} |\Phi_2 \rangle$",color="Tab:blue",alpha=0.8)
            axins.plot(x,Left,label=r"$e^{T} |\Phi_1 \rangle$",color="Tab:green",alpha=0.8)
            if j==0:
                axins.plot(x,CCEVC_energies[i][j],"--",label=r"EVC $|\Phi_1\rangle$",color="tab:orange")
            if j==1:
                axins.plot(x,CCEVC_energies[i][j],"--",label=r"EVC $|\Phi_2\rangle$",color="tab:red")
            mark_inset(axes[i][j], axins, loc1=2, loc2=4, fc="none", ec="0.5")
#axes[1,0].legend(loc="best",handletextpad=0.3,labelspacing = 0.1)
#axes[1,1].legend(loc="best",handletextpad=0.3,labelspacing = 0.1)
handles, labels = axes[0][0].get_legend_handles_labels()
handles2, labels2 = axes[-1][-1].get_legend_handles_labels()

print(handles)
print(labels)
#handles.append(handles2[-3])
handles.append(handles2[-2])
savedh=handles[-3]
labels[-2]=r"EVC $|\Phi_1\rangle$"
labels.append(r"EVC $|\Phi_2\rangle$")
#labels.append(r"Exc.")
savedl=labels[-3]
del labels[-3]; del handles[-3]
labels.append(savedl); handles.append(savedh)
fig.legend(handles, labels, bbox_to_anchor=(0.62,0.55),loc="center",handletextpad=0.3,labelspacing = 0.1)



#plt.plot(x,E_FCI,label="FCI")
#sample_geom1=np.linspace(0,4,81)

#plt.plot(x,Left,label="Left")
#plt.plot(x,Right,label="Right")

#plt.plot(x,E_WFCCEVC,label="pinv")
#plt.plot(x,E_guptri,label="+diag")
#plt.ylim([-15.53,-15.3])
plt.tight_layout()
plt.savefig("resultsandplots/BeH2_insertion_CC.pdf")
plt.show()
#np.savetxt("energy_data/13points_new.txt",E_guptri)
