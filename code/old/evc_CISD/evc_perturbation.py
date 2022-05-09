import pyscf
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, ao2mo
def build_molecule(x,molecule,basis_type):
    """Create a molecule object with parameter x"""

    mol=gto.Mole()
    mol.atom="""%s"""%(molecule(x))
    mol.charge=0
    mol.spin=0
    mol.unit="bohr"
    mol.basis=basis_type
    mol.build()
    return mol
"""
basis="6-31G"
molecule_name="BeH2"
molecule=lambda x: "Be 0 0 0; H %f %f 0; H %f %f 0"%(x,2.54-0.46*x,x,-(2.54-0.46*x))
"""
molecule=lambda x: "H 0 0 0; F 0 0 %f"%x
basis="cc-pVDZ"

def energy_gap(mo_occ,mo_energy):
    HOMO=np.max(np.where(mo_occ>0))
    LUMO=np.min(np.where(mo_occ<=0))
    egap=mo_energy[LUMO]-mo_energy[HOMO]
    return egap
def egap_x(x,coupling_param,molecule,basis):
    mol=build_molecule(x,molecule,basis)
    eri=mol.intor('int2e',aosym="s1")*coupling_param
    mf=mol.RHF()
    mf._eri = ao2mo.restore(1,eri,mol.nao_nr())
    mol.incore_anyway=True
    mf.kernel()
    #mf=scf.addons.convert_to_uhf(mf)
    coefs=mf.mo_coeff
    occ=mf.mo_occ
    energy=mf.mo_energy
    egap=energy_gap(occ,energy)
    return egap

x=np.linspace(1.2,4.5,50)
for coupling_param in np.linspace(-0.2,1.2,7):
    energygap=[egap_x(xval,coupling_param,molecule,basis) for xval in x]
    plt.plot(x,energygap,label="%f"%coupling_param)
plt.legend()
plt.show()
