import numpy as np
from numpy import linalg
from numba import jit
import scipy
import numba as nb
def swap_cols(arr, frm, to): #Swap columns of a matrix
    arrny=arr.copy()
    arrny[:,[frm, to]] = arrny[:,[to, frm]]
    return arrny
def generalized_eigenvector(T,S,symmetric=True):
    """Solves the generalized eigenvector problem.

    Input:
    T: The symmetric matrix
    S: The overlap matrix
    symmetric: Wether the matrices T, S are symmetric

    Returns: The lowest eigenvalue & eigenvector"""
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
    #Compute the first order adjajency matrix
    if detX is None:
        detX=np.linalg.det(X) #Do something
    return detX*linalg.inv(X)
def first_order_adj_matrix_blockdiag(XL,XR,detX=None):
    """Compute the first order adjajency matrix of a two-block matrix, where XL and XR are the left/right matrix, respectively"""
    if detX is None:
        detX=np.linalg.det(XL)*np.linalg.det(XR)
    if detX==0: #Do something
        pass
    return detX*linalg.inv(XL),detX*linalg.inv(XR)
#@jit(nopython=True,parallel=True)
def second_order_compound(X):
    """Compute the second order compound matrix of a matrix X"""
    n=len(X)
    M=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)),dtype=np.float64)
    for j in range(n):
        for i in range(j):
            for l in range(n):
                for k in range(l):
                    M[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=X[i,k]*X[j,l]-X[i,l]*X[j,k]
    return M
#@jit(nopython=True)
def second_order_compound_blockdiag(XL,XR): #XLeft, XRight
    """Compute the second order compound matrix of a block-diagonal matrix """
    n=len(XL)+len(XR)
    na=len(XL) #Alpha/1
    M=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)),dtype=np.float64)
    """

    The following possibilities will yield nonzero elements:
    - All alpha (1/16)
    - All beta (1/16)
    - j beta, i alpha, l beta, k alpha (the way we count)
    """
    for j in range(na):
        for i in range(j):
            for l in range(na):
                for k in range(l):
                    M[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=XL[i,k]*XL[j,l]-XL[i,l]*XL[j,k]
    for j in range(na,n):
        for i in range(na,j):
            for l in range(na,n):
                for k in range(na,l):
                    M[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=XR[i-na,k-na]*XR[j-na,l-na]-XR[i-na,l-na]*XR[j-na,k-na]
    for j in range(na,n):
        for i in range(0,na):
            for l in range(na,n):
                for k in range(0,na):
                    M[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=XL[i,k]*XR[j-na,l-na]#-X[i,l]*X[j,k]
    return M
@jit(nopython=True)
def second_order_compound_blockdiag_separated(XL,XR): #XLeft, XRight
    """Compute the second order compound matrix of a block-diagonal matrix, returning the three relevant blocks """
    n=len(XL)+len(XR)
    na=len(XL) #Alpha/1
    nb=len(XR)
    M1=np.zeros((int(na*(na-1)/2),int(na*(na-1)/2)))
    M2=np.zeros((na*nb,na*nb))
    M3=np.zeros((int(nb*(nb-1)/2),int(nb*(nb-1)/2)))
    """
    The following possibilities will yield nonzero elements:
    - All alpha (1/16)
    - All beta (1/16)
    - j beta, i alpha, l beta, k alpha (the way we count)
    """
    for j in range(na):
        for i in range(j):
            for l in range(na):
                for k in range(l):
                    M1[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=XL[i,k]*XL[j,l]-XL[i,l]*XL[j,k]
    for j in range(na,n):
        for i in range(na,j):
            for l in range(na,n):
                for k in range(na,l):
                    M3[int((j-na)*(j-na-1)/2+i-na),int((l-na)*(l-na-1)/2+k-na)]=XR[i-na,k-na]*XR[j-na,l-na]-XR[i-na,l-na]*XR[j-na,k-na]
    for j in range(nb):
        for i in range(0,na):
            for l in range(nb):
                for k in range(0,na):
                    M2[int(j*na+i),int(l*na+k)]=XL[i,k]*XR[j,l]#-X[i,l]*X[j,k]
    return M1,M2,M3
@jit(nopython=True)
def second_order_compound_blockdiag_separated_RHF(X): #XLeft, XRight
    """Compute the second order compound matrix of a block-diagonal matrix with identical blocks, returning the three relevant blocks - the third block is identical to the first."""
    n=len(X)*2
    nh=len(X) #Alpha/1
    M1=np.zeros((int(nh*(nh-1)/2),int(nh*(nh-1)/2)))
    M2=np.zeros((nh*nh,nh*nh))
    for j in range(nh):
        for i in range(j):
            for l in range(nh):
                for k in range(l):
                    M1[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=X[i,k]*X[j,l]-X[i,l]*X[j,k]
    for j in range(nh):
        for i in range(0,nh):
            for l in range(nh):
                for k in range(0,nh):
                    M2[int(j*nh+i),int(l*nh+k)]=X[i,k]*X[j,l]
    return M1,M2
#@jit(nopython=True)
def second_order_adj_matrix_blockdiag(XL,XR,detX=None):
    if detX is None:
        detX=np.linalg.det(XL)*np.linalg.det(XR)
    first_order=first_order_adj_matrix_blockdiag(XL,XR,detX)
    compund = second_order_compound_blockdiag(first_order[0],first_order[1])
    return 1/detX*compund
def second_order_adj_matrix_blockdiag_separated_RHF(X,detX=None):
    if detX is None:
        detX=np.linalg.det(X)
        detX*=detX
    first_order=first_order_adj_matrix_blockdiag(X,X,detX)
    M1,M2 = second_order_compound_blockdiag_separated_RHF(first_order[0])
    return 1/detX*M1, 1/detX*M2

def second_order_adj_matrix_blockdiag_separated(XL,XR,detX=None):
    if detX is None:
        detX=np.linalg.det(XL)*np.linalg.det(XR)
    first_order=first_order_adj_matrix_blockdiag(XL,XR,detX)
    M1,M2,M3 = second_order_compound_blockdiag_separated(first_order[0],first_order[1])
    return 1/detX*M1, 1/detX*M2,1/detX*M3
def second_order_adj_matrix(X,detX=None):
    if detX is None:
        detX=np.linalg.det(X)
    first_order=first_order_adj_matrix(X,detX)
    return 1/detX*second_order_compound(first_order)
def dot_py(A,B):
    "Dot product between two matrices"
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
def get_antisymm_element(MO_eri,n,na=None,nb=None):
    nh=int(n/2)
    if(na is None or nb is None):
        na=nh
        nb=nh
    G_mat=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)),dtype=np.float64)
    for j in range(na):
        for i in range(j):
            for l in range(na):
                for k in range(l):
                    gleft=MO_eri[i,k,j,l]
                    gright=MO_eri[i,l,j,k]
                    G_mat[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=gleft-gright

    for j in range(na,n):
        for i in range(0,na):
            for l in range(na,n):
                for k in range(0,na):
                    gleft=MO_eri[i,k,j-na,l-na]
                    G_mat[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=gleft
    for j in range(na,n):
        for i in range(na,j):
            for l in range(na,n):
                for k in range(na,l):
                    gleft=MO_eri[i-na,k-na,j-na,l-na]
                    gright=MO_eri[i-na,l-na,j-na,k-na]
                    G_mat[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=gleft-gright
    return G_mat
@jit(nopython=True)
def get_antisymm_element_separated_RHF(MO_eri,n,na=None,nb=None):
    nh=int(n/2)
    if(na is None or nb is None):
        na=nh
        nb=nh
    G1=np.zeros((int(na*(na-1)/2),int(na*(na-1)/2)))
    G2=np.zeros((na*na,nb*nb))
    for j in range(na):
        for i in range(j):
            for l in range(na):
                for k in range(l):
                    gleft=MO_eri[i,k,j,l]
                    gright=MO_eri[i,l,j,k]
                    G1[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=gleft-gright
    for j in range(nb):
        for i in range(na):
            for l in range(nb):
                for k in range(na):
                    gleft=MO_eri[i,k,j,l]
                    G2[int(j*nb+i),int(l*nb+k)]=gleft
    return G1, G2
@jit(nopython=True)
def get_antisymm_element_separated(MO_eriaaaa,Moeribbbb,Moeriaabb,n,na=None,nb=None):
    nh=int(n/2)
    if(na is None or nb is None):
        na=nh
        nb=nh
    G1=np.zeros((int(na*(na-1)/2),int(na*(na-1)/2)))
    G2=np.zeros((na*nb,na*nb))
    G3=np.zeros((int(nb*(nb-1)/2),int(nb*(nb-1)/2)))
    for j in range(na):
        for i in range(j):
            for l in range(na):
                for k in range(l):
                    gleft=MO_eriaaaa[i,k,j,l]
                    gright=MO_eriaaaa[i,l,j,k]
                    G1[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=gleft-gright
    for j in range(na,n):
        for i in range(na,j):
            for l in range(na,n):
                for k in range(na,l):
                    gleft=Moeribbbb[i-na,k-na,j-na,l-na]
                    gright=Moeribbbb[i-na,l-na,j-na,k-na]
                    G3[int((j-na)*(j-na-1)/2+i-na),int((l-na)*(l-na-1)/2+k-na)]=gleft-gright
    for j in range(nb):
        for i in range(na):
            for l in range(nb):
                for k in range(na):
                    gleft=Moeriaabb[i,k,j,l]
                    G2[int(j*na+i),int(l*na+k)]=gleft
    return G1, G2, G3
