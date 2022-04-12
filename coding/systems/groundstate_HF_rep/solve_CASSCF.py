import sys
sys.path.append("../../eigenvectorcontinuation/")

from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
basis="6-31G*"
def molecule(x):
    return "F 0 0 0; H 0 0 %f"%x
molecule_name=r"Hydrogen Fluoride"
xc_array=np.linspace(1.2,5,39)
CASSCF_energies=[]
for i, x in enumerate(xc_array):
    print(i)
    mol =gto.Mole(
        atom = molecule(x),
        basis = basis,
        unit="bohr",
        spin = 0)
    mol.build()
    myhf = mol.RHF().run()
    ncas, nelecas = (6,(2,2))
    mycas = mcscf.CASCI(myhf,6, 4).run()
    CASSCF_energies.append(mycas.e_tot)
    print(mycas.e_tot)
plt.plot(xc_array,CASSCF_energies)
plt.show()
