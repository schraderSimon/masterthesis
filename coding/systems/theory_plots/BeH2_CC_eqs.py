from pyscf import scf,fci,gto,cc,fci
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
from scipy.linalg import fractional_matrix_power
np.set_printoptions(linewidth=300,precision=10,suppress=True)

def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo
    B=mo_new.T
    M=B@A
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0

def localize_procrustes(mol,mo_coeff,mo_occ,ref_mo_coeff,mix_states=False,active_orbitals=None,nelec=None, return_R=False):
    """Performs the orthgogonal procrustes on the occupied and the unoccupied molecular orbitals.
    ref_mo_coeff is the mo_coefs of the reference state.
    If "mix_states" is True, then mixing of occupied and unoccupied MO's is allowed.
    """
    if active_orbitals is None:
        active_orbitals=np.arange(len(mo_coeff))
    if nelec is None:
        nelec=int(np.sum(mo_occ))
    active_orbitals_occ=active_orbitals[:nelec//2]
    active_orbitals_unocc=active_orbitals[nelec//2:]
    mo_coeff_new=mo_coeff.copy()
    if mix_states==False:
        mo=mo_coeff[:,active_orbitals_occ]
        premo=ref_mo_coeff[:,active_orbitals_occ]
        R1,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R1
        mo_unocc=mo_coeff[:,active_orbitals_unocc]
        premo=ref_mo_coeff[:,active_orbitals_unocc]
        R2,scale=orthogonal_procrustes(mo_unocc,premo)
        mo_unocc=mo_unocc@R2


        mo_coeff_new[:,active_orbitals_occ]=np.array(mo)
        mo_coeff_new[:,active_orbitals_unocc]=np.array(mo_unocc)
        R=block_diag(R1,R2)
    elif mix_states==True:
        mo=mo_coeff[:,active_orbitals]
        premo=ref_mo_coeff[:,active_orbitals]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff_new[:,active_orbitals]=np.array(mo)

    if return_R:
        return mo_coeff_new,R
    else:
        return mo_coeff_new

def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
sample_x=np.linspace(0,4,41)
basis="STO-6G"
mo_coeff_min_sample=[]
irreps_sample=[]
occdict1={"A1":6,"B1":0,"B2":0}
occdict2={"A1":4,"B1":2,"B2":0}
occdict3={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]
energies=np.zeros((len(sample_x),3))
energies_CC=np.zeros((len(sample_x),3))
emin=np.zeros(len(sample_x))
emin_CC=np.zeros(len(sample_x))

E_FCI=np.zeros(len(sample_x))
ovlp_eigvals=[]
change_to_ref=[]
atom=molecule(0)
mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
mf = scf.RHF(mol)
e=mf.kernel(verbose=0)
ref_det=mf.mo_coeff
mo_occ=mf.mo_occ
active_space=[0,1,2,3,4,5,6]
nelec=6
for k,x in enumerate(sample_x):
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    overlap_matrix=mol.intor("int1e_ovlp")
    eigvals,eigvecs=np.linalg.eigh(np.linalg.inv(overlap_matrix))
    C_x=fractional_matrix_power(overlap_matrix,-0.5)
    procrustes_C,R=localize_procrustes(mol,C_x,mo_occ,ref_det,mix_states=True,active_orbitals=None,nelec=None, return_R=True)
    U,s,Vt=np.linalg.svd(C_x.T@ref_det)
    print(U)
    ovlp_eigvals.append(eigvals)
    change_to_ref.append((procrustes_C-ref_det).ravel())
plt.plot(sample_x,change_to_ref)
plt.show()
sys.exit(1)

for k,x in enumerate(sample_x):
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mf = scf.RHF(mol)
    e=mf.kernel(verbose=0)
    cisolver = fci.FCI(mf)
    E_FCI[k]=cisolver.kernel(verbose=0)[0]

for k,x in enumerate(sample_x):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    for i in [0,1,2]:
        mf = scf.RHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        mo_coeff_temp.append(mf.mo_coeff)
        mo_en_temp.append(mf.mo_energy)
        energies[k,i]=e
        mycc = cc.CCSD(mf)
        mycc.diis_start_cycle = 10
        mycc.run() # this is UCCSD
        energies_CC[k,i]=mycc.e_tot
    emindex=np.argmin(energies[k,:])
    emin[k]=energies[k,emindex]
    emin_CC[k]=energies_CC[k,emindex]
plt.plot(sample_x,emin,label="HF")
plt.plot(sample_x,emin_CC,label="CCSD")
plt.xlabel("x (Bohr)")
plt.ylabel("E (Hartree)")
plt.legend()
plt.tight_layout()
plt.show()
plt.plot(sample_x,abs(emin_CC-E_FCI),label=r"$|E_{CC}-E_{FCI}|$")
plt.plot(sample_x,abs(emin-E_FCI),label=r"|$E_{CC}-E_{HF}|$")
plt.yscale("log")
plt.xlabel("x (Bohr)")
plt.ylabel("E (Hartree)")
plt.legend()
plt.tight_layout()
plt.show()
