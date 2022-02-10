import numpy as np
import matplotlib.pyplot as plt
from qs_ref import construct_pyscf_system_rhf_ref
from coupled_cluster.rccsd import RCCSD, TDRCCSD
import basis_set_exchange as bse
from pyscf import gto, scf,ao2mo
from coupled_cluster.rccsd import rhs_t
from coupled_cluster.rccsd import energies as rhs_e
from scipy.linalg import eig, sqrtm
import scipy
from opt_einsum import contract
from full_cc import orthonormalize_ts, orthonormalize_ts_choice
from scipy.optimize import minimize, root,newton
import autograd

def get_reference_determinant(molecule_func,refx,basis,charge):
    mol = gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = False
    mol.build(atom=molecule_func(*refx), basis=basis)
    hf = scf.RHF(mol)
    hf.kernel()
    return np.asarray(hf.mo_coeff)
def setUpsamples(sample_x,molecule_func,basis,rhf_mo_ref,mix_states=False):
    t1s=[]
    t2s=[]
    l1s=[]
    l2s=[]
    sample_energies=[]
    Cs=[]
    for x in sample_x:
        system,C = construct_pyscf_system_rhf_ref(
            molecule=molecule_func(*x),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_state=rhf_mo_ref,
            mix_states=mix_states,
            return_C=True
        )
        rccsd = RCCSD(system, verbose=False)
        ground_state_tolerance = 1e-10
        rccsd.compute_ground_state(
            t_kwargs=dict(tol=ground_state_tolerance),
            l_kwargs=dict(tol=ground_state_tolerance),
        )
        t, l = rccsd.get_amplitudes()
        t1s.append(t[0])
        t2s.append(t[1])
        l1s.append(l[0])
        l2s.append(l[1])
        Cs.append(C)
        sample_energies.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
    return t1s,t2s,l1s,l2s,sample_energies,Cs

def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom

def molecule(x):
    val=x
    return "H 0 0 %f; Be 0 0 0; H 0 0 -%f"%(x,x)
def molecule(x):
    return "N 0 0 %f; N 0 0 0"%x
sample_geom1=np.linspace(2,3,4)
#sample_geom1=[2.5,3.0,6.0]
sample_x=[[x] for x in sample_geom1]
basis = 'STO-6G'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
ref_x=[2]
run_cc=True
mol = gto.Mole()
mol.unit = "bohr"
mol.charge = charge
mol.cart = False
mol.build(atom=molecule(2), basis=basis)
hf = scf.RHF(mol)
hf.kernel()
mo_occ=hf.mo_occ
nelec=mol.nelec[0]
print(nelec)

t1s,t2s,l1s,l2s,sample_energies,Cs=setUpsamples(sample_x,molecule,basis,None,mix_states=True)
x_of_interest=np.linspace(2,3,4)
def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo.T
    B=mo_new.T
    M=B@A.T
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


E_approx=[]
for x in x_of_interest:
    system,C_sys = construct_pyscf_system_rhf_ref(
        molecule=molecule(x),
        basis=basis,
        add_spin=False,
        anti_symmetrize=False,
        reference_state=None,
        mix_states=False,
        return_C=True
    )
    ESCF=system.compute_reference_energy().real
    S=np.zeros((len(t1s),len(t1s)))
    H=np.zeros((len(t1s),len(t1s)))
    transformed_u=[]
    transformed_h=[]
    for i, xi in enumerate(t1s):

        for j, xj in enumerate(t1s):
            newMol_i,Rinv=localize_procrustes(mol,C_sys,mo_occ,Cs[i],mix_states=True,return_R=True)
            R=np.linalg.inv(C_sys)@newMol_i
            newMol_j,Uinv=localize_procrustes(mol,C_sys,mo_occ,Cs[j],mix_states=True,return_R=True)
            U=np.linalg.inv(C_sys)@newMol_j
            newBas=U.T@C_sys@R
            transform_matrix=(np.linalg.inv(C_sys.T)@newBas.T)
            print(R-U)
            transform_matrix=U
            energy_basis_1eMO = np.einsum('pi,pq,qj->ij', transform_matrix, system.h , transform_matrix)
            energy_basis_2e_mol_chem=ao2mo.get_mo_eri(system.u,(transform_matrix,transform_matrix,transform_matrix,transform_matrix))
            system.construct_general_orbital_system(anti_symmetrize=True)
            f = system.construct_fock_matrix(energy_basis_1eMO, energy_basis_2e_mol_chem)
            t1_error = rhs_t.compute_t_1_amplitudes(f, energy_basis_2e_mol_chem, t1s[i], t2s[i], system.o, system.v, np)
            t2_error = rhs_t.compute_t_2_amplitudes(f, energy_basis_2e_mol_chem, t1s[i], t2s[i], system.o, system.v, np)
            exp_energy=rhs_e.compute_rccsd_ground_state_energy(f, energy_basis_2e_mol_chem, t1s[i], t2s[i], system.o, system.v, np)+ESCF
            X1=t1s[i]-t1s[j]
            X2=t2s[i]-t2s[j]
            overlap=1+contract("ia,ai->",l1s[j],X1)+0.5*contract("ijab,ai,bj->",l2s[j],X1,X1)+0.25*contract("ijab,abij->",l2s[j],X2)
            S[i,j]=overlap

            H[i,j]=overlap*exp_energy
            extra=contract("ia,ai->",l1s[j],t1_error)+contract("ijab,ai,bj->",l2s[j],X1,t1_error)+0.25*contract("ijab,abij->",l2s[j],t2_error)
            H[i,j]=H[i,j]+extra
    e,cl,c=eig(scipy.linalg.pinv(S,atol=1e-3)@H,left=True)
    idx = np.real(e).argsort()
    e = e[idx]
    c = c[:,idx]
    cl = cl[:,idx]
    E_approx.append(np.real(e[0]))
    print(np.diag(H)-sample_energies)
    S_eig,S_vec=np.linalg.eig(S)
plt.plot(x_of_interest,E_approx)
plt.plot(sample_x,sample_energies,"*")
plt.show()
