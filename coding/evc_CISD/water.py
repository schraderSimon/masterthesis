import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf,ao2mo,lo,ci,fci,cc
import scipy
import sys
from numba import jit
import scipy
sys.path.append("../eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *
from scipy.optimize import minimize, minimize_scalar
np.set_printoptions(linewidth=200,precision=5,suppress=True)
from CISD_solver import *
def make_mol(molecule,x,basis="6-31G"):
    mol=gto.Mole()
    mol.atom=molecule(*x)
    mol.basis = basis
    mol.unit= "Bohr"
    mol.build()
    return mol
def molecule(alpha,r1,r2):
    H2_x=r2*np.cos(alpha)
    H2_y=r2*np.sin(alpha)
    water="O 0 0 0; H %f 0 0; H %f %f 0"%(r1,H2_x,H2_y)
    return water
basis="6-31G"
molecule_name="Water"
basischange=False
x_ref=[105,1.81,1.81]
mol=make_mol(molecule,x_ref,basis)
reference_determinant=create_reference_determinant(mol)
number=3
x_sol=[]
all_determinants=[]
reference_solutions=[]
excitation_operators=[]
mo_coeffs=[]

nonzeros=None
for angle in np.linspace(105,115,number):
    for r1 in np.linspace(1.81,3,number):
        for r2 in np.linspace(1.81,3,number):
            x_sol.append([angle,r1,r2])
x_sol.append([95,1.5,1.5])
x_sol.append([125,4.0,4.0])
x_sample=[[105,1.81,1.81],[107,1.81,1.81],[105,2.0,1.81],[105,1.81,2.0]]
sample_determinants=[]
for index, x in enumerate(x_sample):
    mol=make_mol(molecule,x,basis)
    mf=scf.RHF(mol)
    mf.kernel()
    sample_determinants.append(mf.mo_coeff)
for i, x in enumerate(x_sample):
    mol=make_mol(molecule,x,basis)
    sample_determinants[i]=localize_procrustes(mol,sample_determinants[i],mf.mo_occ,reference_determinant) #Minimize to reference
for i, x in enumerate(x_sample):
    mol=make_mol(molecule,x,basis)
    solver=RHF_CISDsolver(mol,mo_coeff=sample_determinants[i])
    energy,sol=solver.solve_T()
    reference_solutions.append(sol)
energy_refx=[]
energy_HF=[]
energies_EVC=np.zeros((len(x_sol),len(x_sample)))
energies_ref=np.zeros(len(x_sol))
for index, x in enumerate(x_sol):
    mol=make_mol(molecule,x,basis)
    mf=scf.RHF(mol)
    energy_HF.append(mf.kernel())
    all_determinants.append(mf.mo_coeff)
for i, x in enumerate(x_sol):
    mol=make_mol(molecule,x,basis)
    all_determinants[i]=localize_procrustes(mol,all_determinants[i],mf.mo_occ,reference_determinant) #Minimize to reference
for i,x in enumerate(x_sol):
    print(i)
    mol=make_mol(molecule,x,basis)
    solver=RHF_CISDsolver(mol,mo_coeff=all_determinants[i])
    solver.make_T()
    e_corr,sol0=solver.solve_T()
    excitation_operators.append(sol0*np.sign(sol0[0]))
    mo_coeffs.append(solver.mo_coeff)
    for j in range(1,len(reference_solutions)+1):
        e,sol=evc(solver.T,reference_solutions[:j],nonzeros)
        energies_EVC[i,j-1]=e
    energies_ref[i]=e_corr
print("alpha, r1, r2,RHF energy, CISD energy, EVC(1) energy, EVC(6) energy")
for i, positions in enumerate(x_sol):
    print("%5.1f & %2.1f&%2.1f&%10.4f&%10.4f&%15.4f&%15.4f\\\\ \hline"%(positions[0],positions[1],positions[2],energy_HF[i],energies_ref[i],energies_EVC[i,0],energies_EVC[i,-1]))
