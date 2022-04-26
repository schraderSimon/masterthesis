from pyscf import gto, scf
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import sys
from scipy.optimize import linear_sum_assignment, minimize, root,newton
from scipy.linalg import block_diag
np.set_printoptions(linewidth=300,precision=3,suppress=True)
def localize_procrustes(mol,mo_coeff,mo_occ,ref_mo_coeff,mix_states=False,active_orbitals=None,nelec=None, return_R=False,weights=None):
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

def orthogonal_procrustes(mo_new,reference_mo,weights=None):
	"""Solve the orthogonal procrustes problem for the matrix"""
	A=mo_new
	B=reference_mo.copy()
	if weights is not None:
	    B=B@np.diag(weights)
	M=A.T@B
	U,s,Vt=scipy.linalg.svd(M)
	return U@Vt, 0

def similiarize_canonical_orbitals(noons_ref,natorbs_ref,noons,natorbs,nelec,S,Sref):
	"""
	Swaps MO's in coefficient matrix in such a way that the coefficients become analytic w.r.t. the reference
	taking special care of symmetry.
	Input:
	noons_ref (array): Natural occupation numbers of reference OR Fock matrix diagonals
	natorbs_ref (matrix): Natural orbitals of reference OR Canonical orbitals
	Sref (matrix): Overlap matrix of reference
	noons (array): Natural occupation numbers of state of state to be adapted
	natorbs (matrix): Natural orbitals of state to be adapted
	S (matrix): Overlap matrices of state to be adapted

	Returns:
	New natural occupation numbers (new ordering) and natural orbitals
	"""


	pairs_ref=[]
	pairs=[]
	i=0
	#reference
	while i<len(noons_ref): #For each natural orbital
	    if i+1==len(noons_ref):
	        break
	    if abs((noons_ref[i])-(noons_ref[i+1]))<1e-5: #When two natural orbitals have the same natural occupation number
	        pairs_ref.append((i,i+1)) #Add to pair list
	        i+=2
	    else:
	        i+=1
	i=0
	#State to be adapted
	while i<len(noons):
	    if i+1==len(noons):
	        break
	    if abs((noons[i])-(noons[i+1]))<1e-5:
	        pairs.append((i,i+1))
	        i+=2
	    else:
	        i+=1
	#Having found the pairs, I have to "match" them. This is done via...procrustifying :)
	#print("Pairs:")
	fits=np.zeros(len(pairs))
	for i in range(len(pairs)):
	    for j in range(len(pairs_ref)):
	        new_orbs,t=orthogonal_procrustes(natorbs[:,pairs[i]],natorbs_ref[:,pairs_ref[j]]) #Similarize pair coefficients
	        fit=np.linalg.norm(natorbs[:,pairs[i]]@new_orbs-natorbs_ref[:,pairs_ref[j]])
	        fits[j]=fit
	    best_fit=np.argmin(fits)
	    new_orbs,t=orthogonal_procrustes(natorbs[:,pairs[i]],natorbs_ref[:,pairs_ref[best_fit]])
	    natorbs[:,pairs[i]]=natorbs[:,pairs[i]]@new_orbs
	#similarities=C_1^TS_1^(-1/2)^T S_2^(-1/2)C_2
	similarities=natorbs_ref.T@np.real(scipy.linalg.fractional_matrix_power(Sref,0.5))@np.real(scipy.linalg.fractional_matrix_power(S,0.5))@natorbs

	assignment = linear_sum_assignment(-np.abs(similarities))[1] #Find best match (simple as this should be "basically" the identity matrix)
	signs=[]
	for i in range(len(similarities)):
	    signs.append(np.sign(similarities[i,assignment[i]])) # Watch out that MO's keep correct sign
	natorbs=natorbs[:,assignment]*np.array(signs)
	noons=noons[assignment]
	return noons, natorbs


basis = 'cc-pVDZ'
charge = 0
molecule=lambda x:  "H 0 0 %f; O 0 0 0; H 0 0 -%f"%(x,x)
xvals=np.concatenate((np.linspace(1.2,1.45,5),np.linspace(1.45,1.6,16),np.linspace(1.6,2.5,10),np.linspace(2.5,2.7,2),np.linspace(2.8,4.0,20)))
xvals=np.concatenate((np.linspace(1.2,1.45,5),np.linspace(1.45,1.6,16),np.linspace(1.6,2.5,10),np.linspace(2.5,2.7,20),np.linspace(2.8,4.0,20)))
xvals=np.concatenate((np.linspace(1.2,1.45,5),np.linspace(1.45,1.6,16),np.linspace(1.6,2.5,10),np.linspace(2.5,2.7,20),np.linspace(2.8,4.0,20)))
xvals=np.linspace(1.2,5,2)
energies=[]
mocoeffs=[]
overlap_matrices=[]
for x in xvals:
    mol=gto.M(atom=molecule(x),basis=basis,unit="Bohr",symmetry=False)
    mol.build()
    overlap_matrices.append(mol.intor("int1e_ovlp"))
    mf=scf.RHF(mol)
    mf.kernel()
    energies.append(mf.mo_energy[:])
    mocoeffs.append(mf.mo_coeff)
Us=[]
procrustes_Us=[]
"""
for i in range(1,len(mocoeffs)):
	print(xvals[i])
	energies[i],mocoeffs[i]=similiarize_canonical_orbitals(energies[i-1],mocoeffs[i-1],energies[i],mocoeffs[i],10,overlap_matrices[i],overlap_matrices[i-1])
	U=scipy.linalg.fractional_matrix_power(overlap_matrices[i],0.5)@mocoeffs[i]
	Us.append(U)
"""
nelec=5
U0=scipy.linalg.fractional_matrix_power(overlap_matrices[0],0.5)@mocoeffs[0][:,:nelec]
for i in range(1,len(mocoeffs)):
	U=scipy.linalg.fractional_matrix_power(overlap_matrices[i],0.5)@mocoeffs[i][:,:nelec]
	new_U,trash=orthogonal_procrustes(U,U0,weights=None)
	new_U=U@new_U

	#new_C=scipy.linalg.fractional_matrix_power(overlap_matrices[i],-0.5)@new_U
	Us.append(new_U)
	new_C=localize_procrustes(mol,mocoeffs[i],mf.mo_occ,mocoeffs[0],mix_states=False,active_orbitals=None,nelec=None, return_R=False,weights=None)[:,:nelec]
	procrUUerino=scipy.linalg.fractional_matrix_power(overlap_matrices[i],0.5)@new_C
	print(np.real(U0[:,:nelec].T@new_U))
	print(np.real(U0[:,:nelec].T@procrUUerino))
	sys.exit(1)
plt.plot(xvals,energies)
plt.show()
