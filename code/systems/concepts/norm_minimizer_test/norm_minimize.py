sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
sys.path.append("../../eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *
def molecule(x):
    return "F 0 0 0;H 0 0 %f"%x
def calculate_norm_difference(H1_1,H2_1,H1_2,H2_2):
    H1_diff=H1_1-H1_2
    H2_diff=H2_1-H2_2
    val=np.sqrt(2*np.sum(np.ravel(H1_diff)**2)+np.sum(np.ravel(H2_diff)**2))
    return val
def make_spinints_aokjemi(dim,energy_basis_2e_mol_chem,alternating):
    spinints_AO_kjemi=np.zeros((dim,dim,dim,dim))
    for p in range(dim):
    	spinP=int(2*(-(p%2)+0.5))
    	for q in range(dim):
    		spinQ=int(2*(-(q%2)+0.5))
    		for r in range(dim):
    			spinR=int(2*(-(r%2)+0.5))
    			for s in range(dim):
    				spinS=int(2*(-(s%2)+0.5))
    				P=p//2
    				Q=q//2
    				R=r//2
    				S=s//2
    				spinints_AO_kjemi[p,q,r,s]=energy_basis_2e_mol_chem[P,Q,R,S]*alternating[p,q]*alternating[r,s]
    return spinints_AO_kjemi

ref_x=2
basis="6-31G"
mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
ref_det=mf.mo_coeff
dim=len(ref_det)*2
alternating=np.array([[(i+j)%2 for i in range(1,dim+1)] for j in range(dim)])

hcore_ref = np.einsum('pi,pq,qj->ij', ref_det, hcore_ao, ref_det)
nao = mol.nao_nr()
eri_ao = mol.intor('int2e')
eri_ref = ao2mo.incore.full(eri_ao, ref_det)
spinints_AO_kjemi=make_spinints_aokjemi(dim,eri_ref,alternating)
spinints_AO_fysikk=np.transpose(spinints_AO_kjemi,(0,2,1,3))
spinints_AO_fysikk_antisymm_ref=spinints_AO_fysikk-np.transpose(spinints_AO_fysikk,(0,1,3,2))
def ravel_kappa(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]
def unravel_kappa(A,m):
    A_w=np.zeros((m, m))
    A_w[np.triu_indices(m, 1)] = A
    return A_w
def cost_func(kappa_raveled,m,coefficient_matrix,H1_old,H2_old):
    kappa=unravel_kappa(kappa_raveled,m)
    kappa=kappa-kappa.T
    unitary=expm(kappa)
    Cnew=unitary
    H1_new=np.einsum('pi,pq,qj->ij', Cnew, H1_old, Cnew)
    H2_new=ao2mo.incore.full(H2_old, Cnew)
    spinints_AO_kjemi=make_spinints_aokjemi(dim,H2_new,alternating)
    spinints_AO_fysikk=np.transpose(spinints_AO_kjemi,(0,2,1,3))
    spinints_AO_fysikk_antisymm=spinints_AO_fysikk-np.transpose(spinints_AO_fysikk,(0,1,3,2))
    cost=calculate_norm_difference(H1_new,spinints_AO_fysikk_antisymm,hcore_ref,spinints_AO_fysikk_antisymm_ref)
    print(cost)
    return cost

x=4
def make_mol(molecule,x,basis="6-31G"):
    mol=gto.Mole()
    mol.atom=molecule(x)
    mol.basis = basis
    mol.unit= "Bohr"
    mol.build()
    return mol
mol=make_mol(molecule,x)
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
procrustes_orbitals=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_det,mix_states=False)
procrustes_orbitals=basischange(ref_det,mol.intor("int1e_ovlp"),mol.nelec[0])
print(np.linalg.norm(procrustes_orbitals-ref_det))
sys.exit(1)
kappa_0=np.zeros((len(mf.mo_coeff),len(mf.mo_coeff)))
kappa_0_ravel=ravel_kappa(kappa_0)
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
H1_old = np.einsum('pi,pq,qj->ij', procrustes_orbitals, hcore_ao, procrustes_orbitals)
eri_ao = mol.intor('int2e')
H2_old = ao2mo.incore.full(eri_ao, procrustes_orbitals)
result=scipy.optimize.minimize(cost_func, kappa_0_ravel, args=(m,procrustes_orbitals,H1_old,H2_old))
kappa=unravel_kappa(result.x,m)
kappa=kappa-kappa.T
unitary=expm(kappa)
new_orbitals=procrustes_orbitals@unitary
H1_new = np.einsum('pi,pq,qj->ij', new_orbitals, hcore_ao, new_orbitals)
H2_new = np.einsum('pi,pq,qj->ij', new_orbitals, hcore_ao, new_orbitals)

print(unitary@unitary.T)
print("ref")
print(hcore_ref)
print("procrustes")
print(H1_old)
print("new")
print(H1_new)
