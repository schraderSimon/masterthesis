import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, fci, mcscf

def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
basis="STO-3G"
occdict1={"A1":6,"B1":0,"B2":0}
occdict2={"A1":4,"B1":2,"B2":0}
occdict3={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]

xs=np.linspace(0,4,41)
EFCI=np.zeros((len(xs),3))
ECAS=np.zeros((len(xs),3))
for k,x in enumerate(xs):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr',spin=0)
    mo_coeff_temp=[]
    mo_en_temp=[]
    for i in [0]:
        mf = scf.UHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        cisolver = fci.FCI(mf)
        ncas, nelecas = (6,8)
        mycas = mcscf.CASCI(mf, 6, 4)
        e, civec = cisolver.kernel(nelec=mol.nelec)
        print('E = %.12f  2S+1 = %.7f' %(e, cisolver.spin_square(civec, mf.mo_coeff[0].shape[1], mol.nelec)[1]))
        EFCI[k,i]=cisolver.kernel(verbose=0,nelec=(3,3))[0]
        #ECAS[k,i]=mycas.kernel(verbose=0)[0]
    print(EFCI[k,0])
plt.plot(xs,EFCI[:,0],label="FCI 1")
#plt.plot(xs,EFCI[:,1],label="FCI 2")
#plt.plot(xs,EFCI[:,2],label="FCI 3")
#plt.plot(xs,ECAS[:,0],label="CAS 1")
#plt.plot(xs,ECAS[:,1],label="CAS 2")
#plt.plot(xs,ECAS[:,2],label="CAS 3")

plt.legend()
plt.savefig("Problems with BeH2.pdf")
plt.show()
