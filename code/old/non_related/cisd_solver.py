import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf, gto, ao2mo, ci,cc
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
molecule=lambda x: """H 0 0 0; F 0 0 %f"""%x
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
        mol1.symmetry=False
        mol1.build()
        mf=mol1.HF().run(verbose=2) #Solve RHF equations to get overlap
        mycisd=cc.CCSD(mf).run()
        nmo=mycisd.get_nmo()
        print(nmo)
        nocc=mycisd.get_nocc()
        print(nocc)
        e=mycisd.e_tot
        energies.append(e)
    #return np.array(energies), ci.cisd.cisdvec_to_amplitudes(mycisd.ci,nmo,nocc)
    return mycisd.t1, mycisd.t2
x=[1.5]
"""
en,amplitudes=CISD_energy_curve(x,basis,molecule)
print(en)
print(amplitudes[1].shape)
print(amplitudes[2].shape)
print(amplitudes[2][0,1,0,0])
print(amplitudes[2][1,0,0,0])
"""
t1,t2=CISD_energy_curve(x,basis,molecule)
print(t1.shape)
print(t2.shape)
