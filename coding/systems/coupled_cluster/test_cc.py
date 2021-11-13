import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import scipy as sp
from pyscf import gto, scf, cc, ao2mo
import sys
sys.path.append("../../eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *

np.set_printoptions(linewidth=300,precision=2,suppress=True)

def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo.T
    B=mo_new.T
    M=B@A.T
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0
def localize_cholesky(mol,mo_coeff,mo_occ):
    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ>0])
    mo=swappistan(mo)
    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo_unocc=cholesky_coefficientmatrix(mo_coeff[:,mo_occ<=0])
    mo_unocc=swappistan(mo_unocc)
    mo_coeff[:,mo_occ<=0]=np.array(mo_unocc)
    return mo_coeff
def localize_procrustes(mol,mo_coeff,mo_occ,previous_mo_coeff=None):
    mo=mo_coeff[:,mo_occ>0]
    premo=previous_mo_coeff[:,mo_occ>0]
    R,scale=orthogonal_procrustes(mo,premo)
    mo=mo@R

    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo_unocc=mo_coeff[:,mo_occ<=0]
    premo=previous_mo_coeff[:,mo_occ<=0]
    R,scale=orthogonal_procrustes(mo_unocc,premo)
    mo_unocc=mo_unocc@R

    mo_coeff[:,mo_occ<=0]=np.array(mo_unocc)
    #print(mo_unocc)

    return mo_coeff


def make_mol(molecule,x,basis="6-31G"):
    mol=gto.Mole()
    mol.atom=molecule(x)
    mol.basis = basis
    mol.unit= "Bohr"
    mol.build()
    return mol
def create_reference_determinant(mol):
    mf=scf.RHF(mol)
    mf.kernel()
    mo_coeff_converted=localize_cholesky(mol,mf.mo_coeff,mf.mo_occ)
    return mo_coeff_converted

molecule=lambda x: "H 0 0 0; F 0 0 %f"%x
#molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
basis="6-31G"
molecule_name="HF"
x_sol=np.linspace(1.5,4,30)
#x_sol=np.array([1.9,2.5])
ref_x=2.5
energies_ref=np.zeros(len(x_sol))
mol=make_mol(molecule,ref_x,basis)
reference_determinant=create_reference_determinant(mol)
all_determinants=[]
reference_solutions=[]
excitation_operators=[]
mo_coeffs=[]
CC_coefficients=[]
#Step 1: Find the Slater-determinants
mf=scf.RHF(mol)
mf.kernel()
mf.mo_coeff=reference_determinant
mycc=cc.CCSD(mf)
mycc.kernel()
reference_cc=mycc.t2
for index, x in enumerate(x_sol):
    mol=make_mol(molecule,x,basis)
    mf=scf.RHF(mol)
    mf.kernel()
    all_determinants.append(localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,reference_determinant))
    #all_determinants.append(localize_cholesky(mol,mf.mo_coeff,mf.mo_occ))
    mf.mo_coeff=all_determinants[index]
    print(mf.mo_coeff.T@mf.get_fock()@mf.mo_coeff) #Fock matrix in MO basis
    mycc=cc.CCSD(mf)
    mycc.kernel()
    CC_coefficients.append(mycc.t2)
sys.exit(1)
plot_indeces=np.random.randint(5,size=(30,4))
def pick(arr,x):
    return arr[x[0],x[1],x[2],x[3]]
for i in range(len(plot_indeces)):
    plotterino=[]
    for j in range(len(CC_coefficients)):
        print(CC_coefficients[j].shape)
        print(CC_coefficients[j][np.ix_(plot_indeces[0])])
        plotterino.append(pick(CC_coefficients[j],plot_indeces[i]))
    plt.plot(plotterino)
plt.show()
norm_changes=[]
for t2 in CC_coefficients:
    norm_changes.append(norm(t2-reference_cc))
plt.plot(norm_changes)
plt.show()
