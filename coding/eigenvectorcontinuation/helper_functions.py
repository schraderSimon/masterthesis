from pyscf import gto, scf, fci,cc,ao2mo, mp, mcscf
import numpy as np
from numpy import linalg
from numba import jit
import numba as nb
import scipy
from scipy.linalg import lu, qr,svd
def parity(permutation):
    permutation = list(permutation)
    length = len(permutation)
    elements_seen = [False] * length
    cycles = 0
    for index, already_seen in enumerate(elements_seen):
        if already_seen:
            continue
        cycles += 1
        current = index
        while not elements_seen[current]:
            elements_seen[current] = True
            current = permutation[current]
    if((length-cycles) % 2 == 0):
        return 1
    else:
        return -1

def swap_cols(arr, frm, to):
    """Swaps the columns of a 2D-array"""
    arrny=arr.copy()
    arrny[:,[frm, to]] = arrny[:,[to, frm]]
    return arrny
def swap_rows(arr, frm, to):
    """Swaps the rows of a 2D-array"""
    arrny=arr.copy()
    arrny[[frm, to],:] = arrny[[to, frm],:]
    return arrny

def energy_curve_UHF(xvals,basis_type,molecule):
    energies=[]
    for x in xvals:
        mol1=gto.Mole()
        mol1.atom=molecule(x) #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit='AU'
        mol1.spin=0 #Assume closed shell
        mol1.verbose=2
        mol1.build()
        mf=scf.UHF(mol1)
        dm_alpha, dm_beta = mf.get_init_guess()
        dm_beta+=np.random.random_sample(dm_beta.shape)
        dm_beta[:2,:2] = 0
        dm = (dm_alpha,dm_beta)

        energy=mf.kernel(dm)
        energies.append(energy)
    return np.array(energies)
def energy_curve_RHF(xvals,basis_type,molecule):
    energies=[]
    for x in xvals:
        mol1=gto.Mole()
        mol1.atom=molecule(x) #take this as a "basis" assumption.
        mol1.basis=basis_type
        mol1.unit='AU'
        mol1.spin=0 #Assume closed shell
        mol1.verbose=2
        mol1.symmetry=True
        mol1.build()
        mf=scf.RHF(mol1)
        energy=mf.kernel()
        energies.append(energy)
    return np.array(energies)

def CC_energy_curve(xvals,basis_type,molecule):
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
        mf=mol1.RHF().run(verbose=2) #Solve RHF equations to get overlap
        ccsolver=cc.CCSD(mf).run(verbose=2)
        energy=ccsolver.e_tot
        energy+= ccsolver.ccsd_t()
        energies.append(energy)
    return np.array(energies)
def FCI_energy_curve(xvals,basis_type,molecule):
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
        mf=mol1.RHF().run(verbose=2) #Solve RHF equations to get overlap
        cisolver = fci.FCI(mol1, mf.mo_coeff)
        e, fcivec = cisolver.kernel()
        energies.append(e)
    return np.array(energies)
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
    return np.array(energies)

def CASCI_energy_curve(xvals,basis_type,molecule):
    energies=[]
    mol1=gto.Mole()
    mol1.atom=molecule(x) #take this as a "basis" assumption.
    mol1.basis=basis_type
    mol1.unit='AU'
    mol1.symmetry=True
    mol1.spin=0 #Assume closed shell
    myhf = mol.RHF().run()
    # Use MP2 natural orbitals to define the active space for the single-point CAS-CI calculation
    mymp = mp.UMP2(myhf).run()

    noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
    ncas, nelecas = (6,8)
    mycas = mcscf.CASCI(myhf, ncas, nelecas)
    energyies.append(mycas.kernel(natorbs))
