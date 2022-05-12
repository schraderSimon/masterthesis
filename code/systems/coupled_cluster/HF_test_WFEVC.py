import sys
sys.path.append("../libraries")
from func_lib import *
file="energy_data/H_S_test_cc-pVTZ_HF.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
Hs=data["H"]
Ss=data["S"]
CCSD=[-100.14885459358051, -100.23580451413115, -100.2901331075471, -100.32209866524957, -100.33867128519431, -100.34464174181993, -100.34333846786548, -100.33709973295649, -100.32758466402635, -100.31597807003708, -100.30312756844954, -100.28963814526753, -100.27593942701652, -100.26233467331497, -100.2490367649001, -100.23619438795419, -100.22391067723528, -100.21225596387359, -100.20127646754516, -100.19100007700061, -100.18144014288673, -100.1725982203032, -100.1644658919701, -100.15702641685095, -100.15025602946896, -100.14412525485567, -100.13860002850176, -100.13364290959709, -100.12921420689494, -100.12527302774745, -100.12177820607486, -100.11868898808851, -100.11596601394353, -100.1135716040041, -100.11147012081184, -100.10962853397163, -100.10801645655015, -100.1066061195395, -100.10537245472662]
x=np.linspace(1.2,5,39)
E_Schur=[]
E_pseudoinverse=[]
E_SimensMethod=[]
E_guptri=[]
vals=list(np.arange(0,16))
exponent=float(sys.argv[1])
for k in range(len(x)):
    print(k)
    H=np.array(Hs[k][np.ix_(vals,vals)].copy())
    S=np.array(Ss[k][np.ix_(vals,vals)].copy())

    E_pseudoinverse.append(min(np.real(scipy.linalg.eig(scipy.linalg.pinv(S,atol=10**(-8) )  @H)[0])))
    E_SimensMethod.append(min(np.real(scipy.linalg.eig(a=H,b=S+10**(-exponent)*scipy.linalg.expm(-S/10**(-exponent)))[0])))
    E_Schur.append(schur_lowestEigenValue(H,S+10**(-exponent)*scipy.linalg.expm(-S/10**(-exponent))))
    E_guptri.append(guptri_Eigenvalue(H,S,epsu=1e-11,gap=10,zero=False))
plt.plot(x,CCSD,label="CCSD")
#plt.plot(x,E_guptri,label="guptri")
#plt.plot(x,E_pseudoinverse,label="pinv")
plt.plot(x,E_SimensMethod,label="Simen")
plt.plot(x,E_Schur,label="Schur")

plt.legend()
plt.show()
