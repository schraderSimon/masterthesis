from pyscf import gto, scf, ci,lo
import numpy as np
import sys
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
        if matrix[q,q]<1e-14:
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
np.set_printoptions(linewidth=200,precision=5,suppress=True)
mol = gto.M(
    atom = 'Li 0 0 0; H 0 0 1.2',  # in Angstrom
    basis = '6-31G',
)
mf1 = scf.RHF(mol).run() # this is UHF

#mo = lo.ER(mol, mf1.mo_coeff[:,mf1.mo_occ>0])
#mo.init_guess = "Random"
#mo = mo.kernel()
mo=cholesky_coefficientmatrix( mf1.mo_coeff[:,mf1.mo_occ>0])
mo=swappistan(mo)

mf1.mo_coeff[:,mf1.mo_occ>0]=np.array(mo)
mo=cholesky_coefficientmatrix( mf1.mo_coeff[:,mf1.mo_occ<=0])
mo=swappistan(mo)

mf1.mo_coeff[:,mf1.mo_occ<=0]=np.array(mo)
myci1 = ci.CISD(mf1).run() # this is UCISD

ci1=myci1.ci
mol = gto.M(
    atom = 'Li 0 0 0; H 0 0 1.3',  # in Angstrom
    basis = '6-31G',
)
mf2 = scf.RHF(mol).run() # this is UHF
#mf2.mo_coeff=swappistan(mf2.mo_coeff)
mo=cholesky_coefficientmatrix( mf2.mo_coeff[:,mf2.mo_occ>0])
mo=swappistan(mo)

mf2.mo_coeff[:,mf2.mo_occ>0]=np.array(mo)
mo=cholesky_coefficientmatrix( mf2.mo_coeff[:,mf2.mo_occ<=0])
mo=swappistan(mo)

mf2.mo_coeff[:,mf2.mo_occ<=0]=np.array(mo)
myci2 = ci.CISD(mf2).run() # this is UCISD
ci2=myci2.ci
try:
    assert(np.all(np.abs(mf2.mo_coeff-mf1.mo_coeff)<1e-5))
except:
    print(mf2.mo_coeff-mf1.mo_coeff)
    #sys.exit(1)
print("%e"%np.max(np.abs(ci1-ci2)))
