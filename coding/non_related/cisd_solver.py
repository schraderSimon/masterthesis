import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf, gto, ao2mo
import matplotlib.pyplot as plt
def build_molecule(x,basis_type):
    """Create a molecule object with parameter x"""

    mol=gto.Mole()
    mol.atom="""%s"""%(self.molecule(x))
    mol.charge=0
    mol.spin=0
    mol.unit="bohr"
    mol.basis=basis_type
    mol.build()
    return mol
molecule=lambda x: """H 0 0 0; Li 0 0 %f"""%x
basis="6-31G"
def CISD_energy_curve(xvals,basis_type,molecule):
    energies=[]
    for index,x in enumerate(xvals):
        print("%d/%d"%(index,len(xvals)))
        mol1=gto.Mole()
        mol1.atom=molecule(x) #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit='AU'
        mol1.spin=0 #Assume closed shell
        mol1.symmetry=True
        mol1.build()
        mf=mol1.HF().run(verbose=2) #Solve RHF equations to get overlap
        mycisd=mf.CISD().run()
        e=mycisd.e_tot
        energies.append(e)
        plt.plot(mycisd.ci)
        plt.show()
    return np.array(energies)

x=[1.5]
en=CISD_energy_curve(x,basis,molecule)
print(en)
