import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
file="test.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
Hs=data["H"]
Ss=data["S"]
E_WFCCEVC=[]
E_guptri=[]
start=0
stop=30
step=9
file="energy_data/HF_Natorb_0.50.bin"
import pickle
with open(file,"rb") as f:
    data3=pickle.load(f)
xCCSD=data3["xval"]
E_CCSD=data3["CCSD"][1][1]

for i in range(len(Hs)):

    H1=Hs[i][start:stop:step,start:stop:step].copy()
    S1=Ss[i][start:stop:step,start:stop:step].copy()
    H1=Hs[i].copy()
    S1=Ss[i].copy()
    #H1+=np.random.rand(H1.shape[0],H1.shape[1])*1e-16
    eigvals1=np.real(np.linalg.eig(scipy.linalg.pinv(S1,atol=10**(-8))@H1)[0])
    sorted1=np.sort(eigvals1)
    E_WFCCEVC.append(sorted1[0])


    H=Hs[i][start:stop:step,start:stop:step].copy()
    S=Ss[i][start:stop:step,start:stop:step].copy()
    eigvals=np.real(scipy.linalg.eig(a=H,b=S+np.eye(len(S))*10**(-14))[0])
    sorted=np.sort(eigvals)
    print(sorted)
    E_guptri.append(sorted[0])
    #guptrit=guptri_Eigenvalue(H,S,epsu=1e-10,gap=10000,zero=False)
    #E_guptri.append(guptrit)
x=np.linspace(1.2,5.0,len(E_WFCCEVC))
plt.plot(x,E_WFCCEVC,label="pinv")
plt.plot(x,E_guptri,label="+diag")
plt.plot(xCCSD,E_CCSD,label="CCSD")
plt.legend()
plt.show()
#np.savetxt("energy_data/13points_new.txt",E_guptri)
