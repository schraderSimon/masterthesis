from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid

basis = "6-31G*"

charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x
sample_geom_new=[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,5]
sample_geom=[[x] for x in sample_geom_new]
print(sample_geom)
energies=[]
for index,x in enumerate(sample_geom):
    print("%d/%d"%(index,len(sample_geom)))
    mol1=gto.Mole()
    mol1.atom=molecule(*x) #take this as a "basis" assumption.
    mol1.basis=basis
    mol1.unit="bohr"
    mol1.spin=0 #Assume closed shell
    mol1.symmetry=True
    mol1.build()
    mf=mol1.RHF().run(verbose=2) #Solve RHF equations to get overlap
    cisolver = fci.FCI(mol1, mf.mo_coeff)
    e, fcivec = cisolver.kernel()
    energies.append(e)
np.save("HF_1.5to5_FULLCI_631G.npy",energies)
print(energies)
