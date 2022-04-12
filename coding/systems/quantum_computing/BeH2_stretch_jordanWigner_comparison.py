import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from quantum_library import *
def molecule(x):
    return "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
file="energy_data/UCC2_BeH2_stretch.bin"
import pickle
with open(file,"rb") as f:
    dicty=pickle.load(f)
E_FCI=np.array(dicty["E_FCI"])

basis="STO-6G"
dicterino=loadmat("data/BeH2_Jordanwigner_UCCSD2.mat")
print(dicterino)
x_of_interest=dicterino["xvals"][0]
param_list=dicterino["circuits"][0]
UCCSD_energies=dicterino["UCCSD2"][0]
print(UCCSD_energies)
print(x_of_interest)
print(param_list)
plt.plot(x_of_interest,UCCSD_energies,color="black")
plt.plot(x_of_interest,E_FCI)

plt.show()
