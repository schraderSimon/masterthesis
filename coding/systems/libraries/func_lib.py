import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import linalg
from pyscf import gto, scf, mcscf, fci, cc, mp,ao2mo
from guptri_py import *
from scipy.linalg import norm, eig, qz, block_diag, eigh, orth, fractional_matrix_power, expm, svd
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment, minimize, root,newton
from opt_einsum import contract
from scipy.io import loadmat, savemat

import warnings
import sys
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'lines.linewidth': 3})
np.set_printoptions(linewidth=300,precision=3,suppress=True)
def make_mol(molecule,x,basis="6-31G",charge=0):
	"""Helper function to create Mole object at given geometry in given basis"""
	mol=gto.Mole()
	mol.atom=molecule(*x)
	mol.basis = basis
	mol.unit= "Bohr"
	mol.charge=charge
	mol.build()
	return mol
def get_Smat(AO_overlap,HF_coefficients_left,HF_coefficients_right):
	"""Get overlap matrix between two determinants in the same basis at different geometries"""
	print(AO_overlap.shape,HF_coefficients_left.shape,HF_coefficients_right.shape)
	determinant_matrix=contract("ab,ai,bj->ij",AO_overlap,HF_coefficients_left,HF_coefficients_right)
	return determinant_matrix
def localize_cholesky(mol,mo_coeffc,mo_occ):
	"""Obtain cholesky MOs"""
	mo=cholesky_coefficientmatrix(mo_coeffc[:,mo_occ>0])
	mo=swappistan(mo)
	mo_coeffc[:,mo_occ>0]=np.array(mo)
	mo_unocc=cholesky_coefficientmatrix(mo_coeffc[:,mo_occ<=0])
	mo_unocc=swappistan(mo_unocc)
	mo_coeffc[:,mo_occ<=0]=np.array(mo_unocc)
	return mo_coeffc
def basischange(C_old,overlap_AOs_new,neh):
	"""Convert MO's from one set of MOs to a different one using symmetric orthonormalization.
	Input:
	C_old: MO's to convert
	overlap_AOs_new: Overlap matrix at new geometry
	neh: number of electrons divided by two. If this is equal to the number of MOs, this corresponds to general symmetric orthonormalizaton.

	Returns:
	C_new: New set of MOs.
	"""

	C_old=C_old.copy()
	def overlap_p(L,R):
	    return np.einsum("i,j,ij->",L,R,overlap_AOs_new)
	C_occ=C_old[:,:neh]

	S_occ=np.einsum("mi,vj,mv->ij",C_occ,C_occ,overlap_AOs_new)
	S_eig,S_U=np.linalg.eigh(S_occ)
	S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
	C_new_occ=np.einsum("ij,mj->mi",S_powerminusonehalf,C_occ)
	#Remove C_occ part from the unoccupied matrices...

	C_unocc=C_old[:,neh:]
	for unocc_col in range(C_unocc.shape[1]):
	    for occ_col in range(C_new_occ.shape[1]):
	        C_unocc[:,unocc_col]-=C_new_occ[:,occ_col]*overlap_p(C_new_occ[:,occ_col],C_unocc[:,unocc_col])
	S_unocc=np.einsum("mi,vj,mv->ij",C_unocc,C_unocc,overlap_AOs_new)
	S_eig,S_U=np.linalg.eigh(S_unocc)
	S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
	C_new_unocc=np.einsum("ij,mj->mi",S_powerminusonehalf,C_unocc)
	C_new=np.zeros_like(C_old)
	C_new[:,:neh]=C_new_occ
	C_new[:,neh:]=C_new_unocc
	return C_new
def make_mol(molecule,x,basis="6-31G"):
	mol=gto.Mole()
	if isinstance(x,list):
		mol.atom=molecule(*x)
	else:
		mol.atom=molecule(x)
	mol.basis = basis
	mol.unit= "Bohr"
	mol.build()
	return mol
def get_reference_determinant(molecule_func,refx,basis,charge):
    mol = gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = False
    mol.build(atom=molecule_func(*refx), basis=basis)
    hf = scf.RHF(mol)
    hf.kernel()
    return hf.mo_coeff#np.asarray(localize_cholesky(mol,hf.mo_coeff,hf.mo_occ))
def CCSD_energy_curve(molecule_func,xvals,basis):
	"""Returns CCSD energy along a geometry"""
	E=[]
	for x in xvals:
		mol = gto.Mole()
		mol.unit = "bohr"
		mol.build(atom=molecule_func(*x), basis=basis)
		hf = scf.RHF(mol)
		hf.kernel()
		mycc = cc.CCSD(hf)
		mycc.kernel()
		if mycc.converged:
			E.append(mycc.e_tot)
		else:
			E.append(np.nan)
	return E
def orthogonal_procrustes(mo_new,reference_mo,weights=None):
	"""Solve the orthogonal procrustes problem for the matrix"""
	A=mo_new
	B=reference_mo.copy()
	if weights is not None:
	    B=B@np.diag(weights)
	M=A.T@B
	U,s,Vt=scipy.linalg.svd(M)
	return U@Vt, 0
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
def canonical_orthonormalization(T,S,threshold=1e-8):
    """Solves the generalized eigenvector problem.
    Input:
    T: The symmetric matrix
    S: The overlap matrix
    threshold: eigenvalue cutoff
    Returns: The lowest eigenvalue & eigenvector
    """
    ###The purpose of this procedure here is to remove all very small eigenvalues of the overlap matrix for stability
    s, U=np.linalg.eigh(S) #Diagonalize S (overlap matrix, Hermitian by definition)
    U=np.fliplr(U)
    s=s[::-1] #Order from largest to lowest; S is an overlap matrix, hence we (ideally) will only have positive values
    s=s[s>threshold] #Keep only largest eigenvalues
    spowerminushalf=s**(-0.5) #Take s
    snew=np.zeros((len(U),len(spowerminushalf)))
    sold=np.diag(spowerminushalf)
    snew[:len(s),:]=sold
    s=snew


    ###Canonical orthogonalization
    X=U@s
    Tstrek=X.T@T@X
    epsilon, Cstrek = np.linalg.eigh(Tstrek)
    idx = epsilon.argsort()[::1] #Order by size (non-absolute)
    epsilon = epsilon[idx]
    Cstrek = Cstrek[:,idx]
    C=X@Cstrek
    lowest_eigenvalue=epsilon[0]
    lowest_eigenvector=C[:,0]
    return lowest_eigenvalue,lowest_eigenvector

def schur_lowestEigenValue(H,S):
	"""Uses schur decomposition to find lowest eigenvalue of generalized eigenvalue problem"""
	HH, SS, Q, Z = qz(H, S)
	for i in range(len(SS)):
	    if np.abs(SS[i,i])<1e-12:
	        SS[i,i]=1e10
	e=np.diag(HH)/np.diag(SS)
	idx = np.real(e).argsort()
	e = e[idx]
	print(e)
	return np.real(e[0])
def guptri_Eigenvalue(H,S,epsu=1e-8,gap=1000,zero=True):
	"""Use guptri to find lowest eigenvalue of generalized eigenvalue problem"""
	SS, HH, P, Q, kstr = guptri(H,S,zero=zero,epsu=epsu,gap=gap)
	nonzero=np.where(abs(HH[0,:])>1e-5)[0][0]
	SS_reduced=SS[0:len(SS)-nonzero,nonzero:]
	HH_reduced=HH[0:len(SS)-nonzero,nonzero:]
	e=np.diag(SS_reduced)/np.diag(HH_reduced)
	idx = np.real(e).argsort()
	e = e[idx]
	print(np.real(e[0]))
	return np.real(e[0])
	#return kstr[0]
def similiarize_natural_orbitals(noons_ref,natorbs_ref,noons,natorbs,nelec,S,Sref):
	"""
	Swaps MO's in coefficient matrix in such a way that the coefficients become analytic w.r.t. the reference
	taking special care of symmetry.
	Input:
	noons_ref (array): Natural occupation numbers of reference
	natorbs_ref (matrix): Natural orbitals of reference
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
	    if abs(np.log(noons_ref[i])-np.log(noons_ref[i+1]))<1e-6: #When two natural orbitals have the same natural occupation number
	        pairs_ref.append((i,i+1)) #Add to pair list
	        i+=2
	    else:
	        i+=1
	i=0
	#State to be adapted
	while i<len(noons):
	    if i+1==len(noons):
	        break
	    if abs(np.log(noons[i])-np.log(noons[i+1]))<1e-6:
	        pairs.append((i,i+1))
	        i+=2
	    else:
	        i+=1
	#Having found the pairs, I have to "match" them. This is done via...procrustifying :)
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
	pairs=[]
	return noons, natorbs
