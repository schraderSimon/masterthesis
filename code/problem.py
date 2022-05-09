import numpy
import numpy as np
import sys
np.set_printoptions(linewidth=300,precision=3,suppress=True)
from pyscf import gto,symm,scf
def myocc(mf):
    mol = mf.mol
    irrep_id = mol.irrep_id
    so = mol.symm_orb
    orbsym = symm.label_orb_symm(mol, irrep_id, so, mf.mo_coeff)
    doccsym = numpy.array(orbsym)[mf.mo_occ==2]
    soccsym = numpy.array(orbsym)[mf.mo_occ==1]
    for ir,irname in enumerate(mol.irrep_name):
        print('%s, double-occ = %d, single-occ = %d' %(irname, sum(doccsym==ir), sum(soccsym==ir)))
mol=gto.Mole()
mol.atom="benzene.xyz"
mol.atom="Be 0 0 0; H 1 0 1; H 1 0 -1"
mol.symmetry="C2v"
mol.basis="STO-6G"
mol.build()
for s,i,c in zip(mol.irrep_name, mol.irrep_id, mol.symm_orb):
    print(s, i, c.shape)

mf=scf.RHF(mol)
#mf.irrep_nelec = {'A1': 0, 'B1': 4, 'B2':2}


mf.kernel()
Fao = mf.get_fock()
Fmo = mf.mo_coeff.T @ Fao @ mf.mo_coeff
print(Fao)
sys.exit(1)
myocc(mf)
print(mf.get_irrep_nelec())
