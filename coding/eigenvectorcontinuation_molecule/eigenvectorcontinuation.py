import numpy as np
from numpy import linalg
from numba import jit
import numba as nb
def generalized_eigenvector(T,S,symmetric=True):
    "Solves the generalized eigenvector problem."
    "T: The symmetric matrix"
    "S: The overlap matrix"
    s, U=np.linalg.eigh(S)
    """
    s=np.diag(s)
    X=U@np.linalg.inv(np.sqrt(s))
    """
    U=np.fliplr(U)
    s=s[::-1]
    s=s[s>1e-12]
    s=s**(-0.5)
    snew=np.zeros((len(U),len(s)))
    sold=np.diag(s)
    snew[:len(s),:]=sold
    s=snew
    X=U@s
    Tstrek=X.T@T@X
    if symmetric:
        epsilon, Cstrek = np.linalg.eigh(Tstrek)
    else:
        epsilon, Cstrek = np.linalg.eig(Tstrek)
    idx = epsilon.argsort()[::1]
    epsilon = epsilon[idx]
    Cstrek = Cstrek[:,idx]
    C=X@Cstrek

    lowest_eigenvalue=epsilon[0]
    lowest_eigenvector=C[:,0]
    return lowest_eigenvalue,lowest_eigenvector
@jit(nopython=True)
def first_order_adj_matrix(X,detX=None):
    if detX is None:
        detX=np.linalg.det(X)
    return detX*linalg.inv(X)
@jit(nopython=True,parallel=True)
def second_order_compound(X):
    n=len(X)
    M=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)),dtype=np.float64)
    for j in range(n):
        for i in range(j):
            for l in range(n):
                for k in range(l):
                    M[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=X[i,k]*X[j,l]-X[i,l]*X[j,k]
    return M
#@jit(nopython=True)
def second_order_adj_matrix(X,detX=None):
    if detX is None:
        detX=np.linalg.det(X)
    first_order=first_order_adj_matrix(X,detX)
    return 1/detX*second_order_compound(first_order)
def dot_py(A,B):
    m, n = A.shape
    p = B.shape[1]

    C = np.zeros((m,p))

    for i in range(0,m):
        for j in range(0,p):
            for k in range(0,n):
                C[i,j] += A[i,k]*B[k,j]
    return C
dot_nb = nb.jit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:]), nopython = True)(dot_py)
@jit(nopython=True)
def get_antisymm_element(MO_eri,n,nalpha=None,nbeta=None):

    nh=int(n/2)
    if(nalpha is None or nbeta is None):
        nalpha=nh
        nbeta=nh
    G_mat=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)),dtype=np.float64)
    for j in range(n):
        for i in range(j):
            for l in range(n):
                for k in range(l):
                    gleft=0

                    if (i<nalpha and k< nalpha):
                        if (j<nalpha and l< nalpha):
                            gleft=MO_eri[i,k,j,l]
                        elif(j>=nalpha and l>= nalpha):
                            gleft=MO_eri[i,k,j-nalpha,l-nalpha]
                    elif(i>=nalpha and k>= nalpha):
                        if (j<nalpha and l< nalpha):
                            gleft=MO_eri[i-nalpha,k-nalpha,j,l]
                        elif(j>=nalpha and l>= nalpha):
                            gleft=MO_eri[i-nalpha,k-nalpha,j-nalpha,l-nalpha]
                    gright=0
                    if (i<nalpha and l< nalpha):
                        if (j<nalpha and k< nalpha):
                            gright=MO_eri[i,l,j,k]
                        elif(j>=nalpha and k>=nalpha):
                            gright=MO_eri[i,l,j-nalpha,k-nalpha]
                    elif(i>=nalpha and l>=nalpha):
                        if (j<nalpha and k< nalpha):
                            gright=MO_eri[i-nalpha,l-nalpha,j,k]
                        elif(j>=nalpha and k>=nalpha):
                            gright=MO_eri[i-nalpha,l-nalpha,j-nalpha,k-nalpha]
                    G_mat[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=gleft-gright
    return G_mat
