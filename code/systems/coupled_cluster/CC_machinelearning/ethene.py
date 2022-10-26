from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from func_lib import *
from matrix_operations import *
from helper_functions import *
import pickle

def molecule(x):
    C_pos=2.482945+x
    H_pos=3.548545+x
    return "C 0 0 0 ;C 0 0 %f;H 0 1.728121 -1.0656 ;H 0 -1.728121 -1.0656 ;H 0 1.728121 %f;H 0 -1.728121 %f"%(C_pos,H_pos,H_pos)


xvals=np.linspace(-1,2.8,40) #Larger xvals lead to problems. So 2.8 is last resort. xD
energies=[]
for x in xvals:
    atom=molecule(x)
    mol = gto.M(atom=atom,unit="bohr", basis='ccpvtz')
    mf = scf.RHF(mol)
    mf.init_guess='atom'
    e=mf.kernel()
    energies.append(e)
plt.plot(xvals,energies)
plt.show()
# geometric
"""
from pyscf.geomopt.berny_solver import optimize

conv_params = {
    'gradientmax': 1e-6,  # Eh/Bohr
    'gradientrms': 2e-3,  # Eh/Bohr
    'stepmax': 2e-2,      # Bohr
    'steprms': 1.5e-2,    # Bohr
}
mol_eq = optimize(mf, maxsteps=100,**conv_params)
print(mol_eq.atom_coords())
coord=mol_eq.atom_coords()-mol_eq.atom_coords()[0,:]
atom=""
atom_types=["C","C","H","H","H","H"]
for i in range(len(atom_types)):
    atom+="%s %f %f %f ;"%(atom_types[i],coord[i,0],coord[i,1],coord[i,2])
print(atom)
"""
