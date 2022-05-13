import sys
sys.path.append("../libraries")
from rccsd_gs import *

from func_lib import *
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVDZ'
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x
ref_x=[2]
xvals=np.linspace(1.4,4.0,53)
"""Test continuity of Procrustes orbitals"""
mol=make_mol(molecule,ref_x[0],basis)
neh=mol.nelectron//2
mfref=scf.RHF(mol)
mfref.kernel()
ref_coefficientmatrix=mfref.mo_coeff
procrustes_orbitals=[]
for i in range(len(xvals)):
    x=xvals[i]
    mol=make_mol(molecule,x,basis)
    mfnew=scf.RHF(mol)
    mfnew.kernel()
    procrustes_orbitals.append(localize_procrustes(mol,mfnew.mo_coeff.copy(),mfnew.mo_occ,ref_coefficientmatrix).copy())
for i in range(len(xvals)-1):
    assert np.linalg.norm(procrustes_orbitals[i+1]-procrustes_orbitals[i]) < np.linalg.norm(procrustes_orbitals[i])/len(procrustes_orbitals[i]) #Rather arbitrary, but ascertains "small changes"
"""Test continuity of Natural orbitals"""

ref_x=[2]
mol=make_mol(molecule,ref_x[0],basis)
neh=mol.nelectron//2
mfref=scf.RHF(mol)
mfref.kernel()
ref_coefficientmatrix=mfref.mo_coeff

geom_alphas=[[x] for x in xvals]
t1ss,t2s_natorbref,l1ss,l2ss,sample_energiess,reference_natorb_list,reference_overlap_list,reference_noons_list=setUpsamples_naturalOrbitals(geom_alphas,molecule,basis,desired_samples=[6])
natorbs,noons,S=get_natural_orbitals(molecule,geom_alphas,basis,reference_natorb_list[0],reference_noons_list[0],reference_overlap_list[0])
for i in range(len(xvals)-1):
    assert np.linalg.norm(natorbs[i+1]-natorbs[i]) < np.linalg.norm(natorbs[i])/len(natorbs[i]) #Rather arbitrary, but ascertains "small changes"

charge=0


"""Test HF-EVC"""

from REC import *
geom_alphas1=np.linspace(1.500,4,2)
energiesHF=energy_curve_RHF(geom_alphas1,basis,molecule=molecule)
E_CCSD=CC_energy_curve(geom_alphas1,basis,molecule=molecule)
eigvecsolver=eigvecsolver_RHF(geom_alphas1,basis,molecule=molecule,type="procrustes")
energiesEC,eigenvectors=eigvecsolver.calculate_energies(geom_alphas1)
assert np.all((energiesEC-energiesHF)<0), "multi ref. EVC energies higher than HF at sample geometries"
assert np.all((energiesEC-E_CCSD)>0), "multi ref. EVC energies lower than CCSD at sample geometries"
HF=eigvecsolver_RHF_coupling([0.9,1,1.1],geom_alphas1,basis,molecule=molecule,symmetry=True)
energiesEC,eigenvectors=HF.calculate_energies(geom_alphas1)
assert np.all((energiesEC-energiesHF)<0), "lambda-tweaked EVC energies higher than HF at sample geometries"
assert np.all((energiesEC-E_CCSD)>0), "lambda-tweaked EVC energies lower than CCSD at sample geometries"
"""Test WF-CCEVC, AMP-CCEVC and param. reduced AMP-CCEVC"""
reference_determinant=get_reference_determinant(molecule,ref_x,basis,charge)
sample_geometry=np.linspace(1.5,4,2)
geom_alphas1=np.linspace(1.500,1.500,1)
geom_alphas=[[x] for x in geom_alphas1]
sample_geom=[[x] for x in sample_geometry]
E_CCSD=CCSD_energy_curve(molecule,geom_alphas,basis)

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1,start_guess_list=[[3,4]]) #Use wrong start guesses on purpose to test convergence of quasinewton
E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=0.1,start_guess_list=[[3,4]])  #Use wrong start guesses on purpose to test convergence of quasinewton
E_WF=evcsolver.solve_WFCCEVC()
assert np.allclose(E_CCSD,E_AMP_red), "param. reduced AMP-CCEVC with two sample point gives wrong energy at sample point"
assert np.allclose(E_CCSD,E_AMP_full), "AMP-CCEVC with two sample points gives wrong energy at sample point"
assert np.allclose(E_CCSD,E_WF), "WF-CCEVC with two sample points gives wrong energy at sample point" #Not strictly required, but observed in original article
geom_alphas1=np.linspace(1.700,1.700,1)
geom_alphas=[[x] for x in geom_alphas1]
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1,start_guess_list=[[3,4]]) #Use wrong start guesses on purpose
E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=0.1,start_guess_list=[[3,4]])  #Use wrong start guesses on purpose
E_WF=evcsolver.solve_WFCCEVC()
assert (not np.allclose(E_CCSD,E_AMP_red)), "param. red. AMP-CCEVC gives CCSD energy where it shouldn't"
assert (not np.allclose(E_CCSD,E_AMP_full)), "AMP-CCEVC gives CCSD energy where it shouldn't"
assert (not np.allclose(E_CCSD,E_WF)), "WF-CCEVC gives CCSD energy where it shouldn't"
