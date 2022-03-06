import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *

def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
"""
def molecule(x):
    return "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
"""
xs=np.linspace(2,3.5,46)
basis="STO-6G"
eigvals=[]
energies=[]
for x in xs:
    mol=gto.M(atom=molecule(x),basis=basis,spin=0,unit="bohr",symmetry="c2v")

    myhf=scf.RHF(mol)
    e=myhf.run()
    ncas, nelecas = (len(myhf.mo_coeff),mol.nelectron)
    print(ncas,nelecas)
    mc = mcscf.CASCI(myhf, ncas, nelecas)
    res = mc.kernel()
    print(res[0])
    rep=linalg.fractional_matrix_power(mol.intor("int1e_ovlp"), 0.5)@mc.make_rdm1()@linalg.fractional_matrix_power(mol.intor("int1e_ovlp"), 0.5)
    rep=rep[3:,3:]
    natocc = np.linalg.eigh(rep)[0]
    print(rep)
    eigvals.append(natocc)
    print(natocc,np.sum(natocc))
    energies.append(res[0])
plt.plot(xs,eigvals)
plt.yscale("log")
plt.show()
