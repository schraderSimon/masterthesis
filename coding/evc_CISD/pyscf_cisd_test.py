from pyscf import gto, scf, ci,lo
import numpy as np
import sys
from numba import jit
import matplotlib.pyplot as plt
def swap_cols(arr, frm, to):
    """Swaps the columns of a 2D-array"""
    arrny=arr.copy()
    arrny[:,[frm, to]] = arrny[:,[to, frm]]
    return arrny
def swappistan(matrix):
    swapperinos=[]
    for i in range((matrix.shape[1])):
        for j in range(i+1,matrix.shape[1]):
            sort_i=np.sort(matrix[:,i])
            sort_j=np.sort(matrix[:,j])
            if(np.all(np.abs(sort_i-sort_j)<1e-8)): #If the two columns are equal
                nonzero_i=np.where(np.abs(matrix[:,i])>=1e-5)[0][0]
                nonzero_j=np.where(np.abs(matrix[:,j])>=1e-5)[0][0]
                if nonzero_i>nonzero_j:
                    matrix=swap_cols(matrix,i,j)
    return matrix

def cholesky_pivoting(matrix):
    n=len(matrix)
    R=np.zeros((n,n))
    piv=np.arange(n)
    for k in range(n):
        q=np.argmax(np.diag(matrix)[k:])+k
        if matrix[q,q]<1e-10:
            break
        temp=matrix[:,k].copy()
        matrix[:,k]=matrix[:,q]
        matrix[:,q]=temp
        temp=R[:,k].copy()
        R[:,k]=R[:,q]
        R[:,q]=temp
        temp=matrix[k,:].copy()
        matrix[k,:]=matrix[q,:]
        matrix[q,:]=temp
        temp=piv[k]
        piv[k]=piv[q]
        piv[q]=temp
        R[k,k]=np.sqrt(matrix[k,k])
        R[k,k+1:]=matrix[k,k+1:]/R[k,k]
        matrix[k+1:n,k+1:n]=matrix[k+1:n,k+1:n]-np.outer(R[k,k+1:],R[k,k+1:])
    P=np.eye(n)[:,piv]
    return R,P

def cholesky_coefficientmatrix(matrix):
    D=2*matrix@matrix.T
    R,P=cholesky_pivoting(D)
    PL=P@R.T
    Cnew=PL[:,:matrix.shape[1]]/np.sqrt(2)
    return Cnew
    for i in range(Cnew.shape[1]):
        if np.sum(Cnew[:,i])<0:
            Cnew[:,i]=-Cnew[:,i]
def localize_mocoeff(mol,mo_coeff,mo_occ):
    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ>0])
    mo=swappistan(mo)
    #monew=lo.ER(mol,mo)
    #monew.init_guess=mo
    #mo=monew.kernel()
    mo_coeff[:,mo_occ>0]=np.array(mo)
    mo=cholesky_coefficientmatrix(mo_coeff[:,mo_occ<=0])
    mo=swappistan(mo)
    #monew=lo.ER(mol,mo)
    #monew.init_guess=mo
    #mo=monew.kernel()
    mo_coeff[:,mo_occ<=0]=np.array(mo)

    return mo_coeff

np.set_printoptions(linewidth=200,precision=5,suppress=True)
def molfunc(x):
    return gto.M(atom = 'Li 0 0 0; H 0 0 %f'%x,unit="Bohr",basis = '6-31G')
xarray=np.linspace(1,6,50)
orbital_energies=[]
PL_matrices=[]
CI_coefficients=[]
coefficients_selector=np.random.choice(150,replace=False,size=20);#[1,10,25,33,60,85,100,123,145] #More or less randomized numbers
def basischange(C_old,overlap_AOs_newnew):
    S_eig,S_U=np.linalg.eigh(overlap_AOs_newnew)
    S_poweronehalf=S_U@np.diag(S_eig**0.5)@S_U.T
    S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
    C_newbasis=S_poweronehalf@C_old #Basis change
    q,r=np.linalg.qr(C_newbasis) #orthonormalise
    return S_powerminusonehalf@q #change back
mol=molfunc(3)
mf=scf.RHF(mol).run()
mo_coeff_0=mf.mo_coeff

for xval in xarray:
    mol=molfunc(xval)
    mf = scf.RHF(mol).run()
    energy_mf=mf.mo_energy
    #PL=localize_mocoeff(mol,mf.mo_coeff,mf.mo_occ)
    PL=basischange(mo_coeff_0,mol.intor("int1e_ovlp"))
    orbital_energies.append(energy_mf)

    mf.mo_coeff=PL
    PL_matrices.append(mf.mo_coeff)
    myci=ci.CISD(mf).run()
    CI_coefficients.append(myci.ci[coefficients_selector])

PL_matrices_diff=[]
vector_norms=[]
for i in range(len(xarray)-1):
    PL_matrices_diff.append(PL_matrices[i+1]-PL_matrices[i])
    vector_norms.append(np.linalg.norm(PL_matrices_diff[-1],axis=0))
vector_norms=np.array(vector_norms)
plt.plot(xarray,orbital_energies)
plt.show()
for i in range((vector_norms.shape[1])):
    plt.plot(xarray[:-1],vector_norms[:,i],label="%d"%i)
plt.legend()
plt.show()
plt.plot(xarray,CI_coefficients)
plt.show()
