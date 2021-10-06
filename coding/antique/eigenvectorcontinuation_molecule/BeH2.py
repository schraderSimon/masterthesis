import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, fci,cc,ao2mo, mp, mcscf,symm
import pyscf
import sys
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200,precision=4,suppress=True)

from ec_HF_basischange import *

def myocc(mf):
    mol = mf.mol
    irrep_id = mol.irrep_id
    so = mol.symm_orb
    orbsym = symm.label_orb_symm(mol, irrep_id, so, mf.mo_coeff)
    doccsym = np.array(orbsym)[mf.mo_occ==2]
    soccsym = np.array(orbsym)[mf.mo_occ==1]
    occdict = {}
    print('MO shape = ', mf.mo_coeff.shape)
    for ir,irname in enumerate(mol.irrep_name):
        print('%s, double-occ = %d, single-occ = %d' %
            (irname, sum(doccsym==ir), sum(soccsym==ir)))
        occdict[irname] = 2*sum(doccsym==ir) # THIS IGNORES SINGLY OCC! OK FOR RHF.
    return(occdict)
def get_geometry(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"

    print(atom)
    return atom
molecule_name="BeH2"
Evangelista_basis = """
Be S
1267.070000 0.001940
190.356000 0.014786
43.295900 0.071795
12.144200 0.236348
3.809230 0.471763
1.268470 0.355183
Be S
5.693880 -0.028876
1.555630 -0.177565
0.171855 1.071630
Be S
0.057181 1.000000
Be P
5.693880 1.000000
Be P
1.555630 0.144045
0.171855 0.949692
H S
19.24060 0.032828
2.899200 0.231208
0.653400 0.817238
H S
0.177600 1.000000"""

#my_basis = 'cc-pVDZ'
basis_type = gto.basis.parse(Evangelista_basis)
#basis_type="cc-pVDZ"
sample_x=np.linspace(0,4,5)
xc_array=np.linspace(0,4,41)
molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
mol=gto.Mole()
x=0.1
mol.atom="""%s"""%(molecule(x))
mol.charge=0
mol.spin=0
mol.symmetry="c2v"
mol.unit="AU"
mol.basis=basis_type
mol.build()
mf=scf.RHF(mol)
mf.kernel()
dm_init_guess_1 = mf.make_rdm1()

occdict=myocc(mf)
print(occdict)
mol=gto.Mole()
x=4
mol.atom="""%s"""%(molecule(x))
mol.charge=0
mol.spin=0
mol.symmetry="c2v"
mol.unit="bohr"
mol.basis=basis_type
mol.build()
mf=scf.RHF(mol)
mf.kernel()
occdict2=myocc(mf)
print(occdict2)

x_range = np.linspace(0.1, 2, 5)
E_HF = np.zeros((len(x_range),2))
occs = [occdict, occdict2]
HFc=[]
xvals=np.linspace(0,4,50)
for i,x in enumerate(x_range):
    for p in range(2):
        print('x = ', x)
        atom = molecule(x)
        mol = gto.M(atom=atom, basis=basis_type, symmetry="C2v", unit='bohr')
        mf = scf.RHF(mol)
        mf.irrep_nelec = occs[p]
        mf.initial_guess="atom"
        E_HF[i,p]=mf.kernel()
        if p==0:
            HFc.append(mf.mo_coeff[:, mf.mo_occ >=-1])
    print(np.array(HFc))
    HF=eigensolver_RHF_knowncoefficients(np.array(HFc),basis_type,molecule=molecule)
    energiesEC,eigenvectors=HF.calculate_energies(xvals)
    plt.plot(xvals,energiesEC,label="%d"%i)
plt.legend()
plt.show()
