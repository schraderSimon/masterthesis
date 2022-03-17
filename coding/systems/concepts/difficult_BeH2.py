import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *

def molecule(x):
    y = lambda x: 1.8 - 0.36*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; O 0 0 0"
    return atom
xvals=np.linspace(0,4,21)
occdict1={"A1":6,"B1":2,"B2":2}
occdict2={"A1":8,"B1":2,"B2":0}
occdict3={"A1":8,"B1":0,"B2":2}
occdict4={"A1":4,"B1":4,"B2":2}
occdict5={"A1":6,"B1":4,"B2":0}
occdict6={"A1":6,"B1":0,"B2":4}

occdicts=[occdict1,occdict2,occdict3,occdict4,occdict5,occdict6]
energies=np.zeros((len(xvals),len(occdicts)))
irreps=[]
mo_coeff_min=[]
basis="6-31G"
for k,x in enumerate(xvals):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    for i in range(len(occdicts)):
        mf = scf.RHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        mo_coeff_temp.append(mf.mo_coeff)
        mo_en_temp.append(mf.mo_energy)
        energies[k,i]=e
    emindex=np.argmin(energies[k,:])
    irreps.append(occdicts[emindex])
    mo_coeff_min.append(mo_coeff_temp[emindex])
plt.plot(xvals,energies)

plt.show()
