import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf, gto
from scipy.optimize import minimize
np.set_printoptions(linewidth=200,precision=5,suppress=True)
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

np.set_printoptions(linewidth=200,precision=5,suppress=True)

def localize_mocoeff(mol,mo_coeff,mo_occ,previous_mo_coeff=None):

    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ>0])
    mo=swappistan(mo)
    if previous_mo_coeff is not None:
        previous_mo=previous_mo_coeff[:,mo_occ>0]
        diff=mo-previous_mo
        """Do the magic and swap columns and rows"""
        col_norm=np.sum(np.abs(diff),axis=0)
        colnorm_descendingargs=np.argsort(col_norm)[::-1]
        colnorm_sorted=col_norm[colnorm_descendingargs]
        if colnorm_sorted[0]>0.9:
            mo=swap_cols(mo,colnorm_descendingargs[0],colnorm_descendingargs[1])

    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ<=0])
    mo=swappistan(mo)
    if previous_mo_coeff is not None:
        previous_mo=previous_mo_coeff[:,mo_occ<=0]
        diff=mo-previous_mo
        """Do the magic and swap columns and rows"""
        col_norm=np.sum(np.abs(diff),axis=0)
        #print(np.abs(diff))
        #print(col_norm)
        colnorm_descendingargs=np.argsort(col_norm)[::-1]
        colnorm_sorted=col_norm[colnorm_descendingargs]
        if colnorm_sorted[0]>1:
            mo=swap_cols(mo,colnorm_descendingargs[0],colnorm_descendingargs[1])
    mo_coeff[:,mo_occ<=0]=np.array(mo)

    return mo_coeff


def basischange(C_old,overlap_AOs_newnew):
    S=np.einsum("mi,vj,mv->ij",C_old,C_old,overlap_AOs_newnew)
    S_eig,S_U=np.linalg.eigh(S)
    S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
    C_new=np.einsum("ij,mj->mi",S_powerminusonehalf,C_old)
    return C_new
basis="6-31G"

H_1=gto.Mole()
H_1.atom="H 0 0 0; Li 0 0 3.2"
H_1.basis=basis
H_1.build()
H_2=gto.Mole()
H_2.atom="H 0 0 0; Li 0 0 3.7"
H_2.basis=basis
H_2.build()

mf1=scf.RHF(H_1)
mf1.kernel()
MO_1=localize_mocoeff(H_1,mf1.mo_coeff,mf1.mo_occ)
MO_1=mf1.mo_coeff
mf2=scf.RHF(H_2)
mf2.kernel()
MO_2=localize_mocoeff(H_2,mf2.mo_coeff,mf2.mo_occ)
MO_2=mf2.mo_coeff
print(np.sum(np.abs(MO_2-MO_1)**2))

neh=H_1.nelectron//2
num_bas=(mf1.mo_coeff.shape[0])
n_unocc=num_bas-neh

def orbital_dissimilarity(alpha,y1,y2,x1,x2):
    ca=np.cos(alpha)
    sa=np.sin(alpha)
    first=np.sum((ca*y1-sa*y2-x1)**2)
    second=np.sum((sa*y1+ca*y2-x2)**2)
    return first+second
def orbital_dissimilarity_dev(alpha,y1,y2,x1,x2):
    ca=np.cos(alpha)
    sa=np.sin(alpha)
    first=np.sum((ca*y1-sa*y2-x1)*(-sa*y1-ca*y2))
    second=np.sum((sa*y1+ca*y2-x2)*((ca*y1-sa*y2)))
    return first+second
for i in range(neh,num_bas):
    for j in range(i+1,num_bas):
        x1=MO_1[:,i]
        x2=MO_1[:,j]
        y1=MO_2[:,i]
        y2=MO_2[:,j]
        alpha=minimize(function_to_minimalize,0,args=(y1,y2,x1,x2),jac=derivative_of_function,method="BFGS").x[0]#,jac=derivative_of_function).x[0]
        y2_new=np.sin(alpha)*MO_2[:,i]+np.cos(alpha)*MO_2[:,j]
        y1_new=np.cos(alpha)*MO_2[:,i]-np.sin(alpha)*MO_2[:,j]
        MO_2[:,i]=y1_new
        MO_2[:,j]=y2_new
        print(i,j,alpha)
print(MO_2-MO_1)
print(np.sum(np.abs(MO_2-MO_1)**2))
print(MO_2.T@H_2.intor("int1e_ovlp")@MO_2) #All we do is change the molecular orbitals. So WHY is this MOTHERFUCKER not orthgogonal??
