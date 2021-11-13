import numpy as np
from numpy import linalg
from numba import jit
import numba as nb
import scipy
from scipy.linalg import lu, qr,svd, det
from helper_functions import *
def swappistan(matrix):
    #return matrix
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

@jit
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
@jit
def cholesky_coefficientmatrix(matrix):
    D=2*matrix@matrix.T
    R,P=cholesky_pivoting(D)
    PL=P@R.T
    Cnew=PL[:,:matrix.shape[1]]/np.sqrt(2)
    return Cnew
def LDU_decomp(X,overwrite_a=True,check_finite=False):
    """Singular-Value-based LDU decomposition.
    Input:
        -The matrix X to decompose
    Returns:
        - (L.T): 2D array of the left matrix
        - d : 1D diagonal matrix
        - R.T: 2D array of the right matrix
        """
    U,s,Vh=svd(X,overwrite_a=overwrite_a,check_finite=check_finite)
    detU=det(U,check_finite=check_finite)
    detVh=det(Vh,check_finite=check_finite)
    if detU<0:
        U[:,0]=-1*U[:,0]
        s[0]=-s[0]
    if detVh<0:
        Vh[0,:]=-1*Vh[0,:]
        s[0]=-s[0]
    return U,s,Vh
def generalized_eigenvector(T,S,symmetric=True):
    """Solves the generalized eigenvector problem.
    Input:
    T: The symmetric matrix
    S: The overlap matrix
    symmetric: Wether the matrices T, S are symmetric
    Returns: The lowest eigenvalue & eigenvector
    """
    ###The purpose of this procedure here is to remove all very small eigenvalues of the overlap matrix for stability
    s, U=np.linalg.eigh(S) #Diagonalize S (overlap matrix, Hermitian by definition)
    U=np.fliplr(U)
    s=s[::-1] #Order from largest to lowest; S is an overlap matrix, hence we won't
    s=s[s>1e-8] #Keep only largest eigenvalues
    spowerminushalf=s**(-0.5) #Take s
    snew=np.zeros((len(U),len(spowerminushalf)))
    sold=np.diag(spowerminushalf)
    snew[:len(s),:]=sold
    s=snew


    ###Canonical orthogonalization
    X=U@s
    Tstrek=X.T@T@X
    if symmetric:
        epsilon, Cstrek = np.linalg.eigh(Tstrek)
    else:
        epsilon, Cstrek = np.linalg.eig(Tstrek)
    idx = epsilon.argsort()[::1] #Order by size (non-absolute)
    epsilon = epsilon[idx]
    Cstrek = Cstrek[:,idx]
    C=X@Cstrek
    lowest_eigenvalue=epsilon[0]
    lowest_eigenvector=C[:,0]
    return lowest_eigenvalue,lowest_eigenvector
def calculate_vector_norm(S,vector):
    norm = np.einsum("ab,a,b->",S,vector,vector)**2
    return norm
def cofactor_index(X):
    """
    Returns the index of the first nonzero cofactor of a matrix X.
    Input:
        The matrix X (which is assumed to have nullity 0 or lower)
    Returns:
        The first index encountered in an element-wise search with a non-zero cofactor.
    """
    C = np.zeros(X.shape)
    nrows, ncols = C.shape
    rows_val=np.arange(X.shape[0])
    cols_val=np.arange(X.shape[1])
    for row in rows_val:
        relrows=rows_val[rows_val!=row]
        for col in cols_val:
            relcols=cols_val[cols_val!=col]
            minor = X[np.ix_(relrows,relcols)]
            C = np.linalg.det(minor)
            if np.abs(C)>1e-7: #Needs to be "relatively large" to assure that matrix later is invertible
                return row,col
    return None
def first_order_adj_matrix_LdR(L,d,R):
    determinant=np.prod(d)
    return R@np.diag(d**(-1))@L*determinant
def first_order_adj_matrix(X,detX=None):
    """Compute the first order adjajency matrix of an invertible matrix"""
    if detX is None:
        detX=np.linalg.det(X)
    try:
        adj_matrix=detX*linalg.inv(X)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Cannot calculate first order adjajency matrix as input is non-invertible.")
    return adj_matrix
def first_order_adj_matrix_blockdiag(XL,XR,detX=None):
    """Compute the first order adjajency matrix of a two-block matrix, where XL and XR are the left/right matrix, respectively"""
    if detX is None:
        detX=np.linalg.det(XL)*np.linalg.det(XR)
    try:
        adj_matrix_left,adj_matrix_right=detX*linalg.inv(XL),detX*linalg.inv(XR)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Cannot calculate first order adjajency matrix as input is non-invertible.")
    return adj_matrix_left,adj_matrix_right
@jit(nopython=True)
def second_order_compound_row(X,j,i):
    """Returns only the row with index(ij) of the second order compound of a matrix X"""
    n=len(X)
    M=np.zeros(int(n*(n-1)/2))
    for l in range(n):
        for k in range(l):
            M[int((l)*(l-1)/2+k)]=X[i,k]*X[j,l]-X[i,l]*X[j,k]
    return M
@jit(nopython=True)
def second_order_compound_col(X,l,k):
    """Returns only the col with index (kl) of the second order compound of a matrix X"""
    n=len(X)
    M=np.zeros(int(n*(n-1)/2))
    for j in range(n):
        for i in range(j):
            M[int((j)*(j-1)/2+i)]=X[i,k]*X[j,l]-X[i,l]*X[j,k]
    return M
@jit(nopython=True)
def second_order_compound(X):
    """Compute the second order compound matrix of a matrix X"""
    n=len(X)
    M=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)),dtype=np.float64)
    for j in range(n):
        for i in range(j):
            for l in range(n):
                for k in range(l):
                    M[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=X[i,k]*X[j,l]-X[i,l]*X[j,k]
@jit(nopython=True)
def second_order_compound_blockdiag(XL,XR): #XLeft, XRight
    """Compute the second order compound matrix of a block-diagonal matrix """
    n=len(XL)+len(XR)
    na=len(XL)
    M=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)),dtype=np.float64)
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
def second_order_adj_matrix_blockdiag(XL,XR,detX=None):
    if detX is None:
        detX=np.linalg.det(XL)*np.linalg.det(XR)
    if np.abs(detX)<1e-20:
        print("Cannot calculate second order adjajency matrix as determinant is zero (Breakdown of Jacobi's theorem).")
        #raise np.linalg.LinAlgError("Cannot calculate second order adjajency matrix as determinant is zero (Breakdown of Jacobi's theorem).")
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
    if np.abs(detX)<1e-20:
        print("Cannot calculate second order adjajency matrix as determinant is zero (Breakdown of Jacobi's theorem).")

        #raise np.linalg.LinAlgError("Cannot calculate second order adjajency matrix as determinant is zero (Breakdown of Jacobi's theorem).")
    det_inv=1/detX
    first_order=first_order_adj_matrix_blockdiag(XL,XR,detX)
    M1,M2,M3 = second_order_compound_blockdiag_separated(first_order[0],first_order[1])
    return det_inv*M1, det_inv*M2,det_inv*M3
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
@jit(nopython=True)
def get_antisymm_element_full(MO_eriaaaa,Moeribbbb,Moeriaabb,Moeribbaa,n,na=None,nb=None): #This should never be used, its simply for simplicity reasons
    nh=int(n/2)
    if(na is None or nb is None):
        na=nh
        nb=nh
    G_mat=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)),dtype=np.float64)
    for j in range(n):
        for i in range(j):
            for l in range(n):
                for k in range(l):
                    gleft=0
                    if (i<na and k< na):
                        if (j<na and l< na):
                            gleft=MO_eriaaaa[i,k,j,l]
                        elif(j>=na and l>= na):
                            gleft=Moeriaabb[i,k,j-na,l-na]
                    elif(i>=na and k>= na):
                        if (j<na and l< na):
                            gleft=Moeribbaa[i-na,k-na,j,l]
                        elif(j>=na and l>= na):
                            gleft=Moeribbbb[i-na,k-na,j-na,l-na]
                    gright=0
                    if (i<na and l< na):
                        if (j<na and k< na):
                            gright=MO_eriaaaa[i,l,j,k]
                        elif(j>=na and k>=na):
                            gright=Moeriaabb[i,l,j-na,k-na]
                    elif(i>=na and l>=na):
                        if (j<na and k< na):
                            gright=Moeribbaa[i-na,l-na,j,k]
                        elif(j>=na and k>=na):
                            gright=Moeribbbb[i-na,l-na,j-na,k-na]
                    G_mat[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=gleft-gright
    return G_mat
def LDU_decomp_new(X,threshold=1e-10):
    P,L,U=lu(X,check_finite=False)
    #P.T @ X = L U
    L=P@L
    #X=L' U
    d=np.diag(U) #Diagonal matrix
    U=np.divide(U.T,d).T
    U[np.isnan(U)]=0
    U[np.isinf(U)]=0

    permute=np.argsort(-d)
    d_new=d[permute]
    U=U[np.ix_(permute,np.arange(len(U)))]
    U[0,:]=U[0,:]*parity(permute)
    d_new[0]=d_new[0]*parity(permute)
    if np.linalg.det(P)<0:
        d_new[0]=d[0]*(-1)
        L[:,-1]=L[:,-1]*(-1)
    detL=np.linalg.det(L)
    detU=np.linalg.det(U)
    detX=np.linalg.det(X)
    detD=np.linalg.det(np.diag(d))
    assert np.abs(detL-1)<1e-10, "detL has wrong determinant"
    assert np.abs(detU-1)<1e-10, "detU has wrong determinant"
    assert np.abs(detX-detD)<1e-10, "detD has wrong determinant"
    print("X")
    print(X)
    print("ShouldBeX")
    print(L@np.diag(d_new)@U)
    print("L")
    print(L)
    print("D")
    print(np.diag(d_new))
    print("U")
    print(U)
    assert np.all(np.abs(L@np.diag(d_new)@U)<1e-10), "not the same matrix"
    return L, d_new, U
def LDU_decomp_fucked(X,threshold=1e-10):
    """Implement the LDU decomposition of a square matrix X"""
    Linv,U,P=qr(X,pivoting=True)
    d=np.diag(U) #Diagonal matrix
    d=d.copy()
    #d[abs(d)<threshold]=0 #ignore that very part of the matrix
    R_inv=np.divide(U.T,d).T

    R_inv[np.isnan(R_inv)]=0
    R_inv[np.isinf(R_inv)]=0
    return Linv, d, R_inv, P
def second_order_compound_row_separated(XL,XR,j,i):
    n=len(XL)+len(XR)
    na=nb=nh=len(XL)
    M=np.zeros(int(n*(n-1)/2))
    M1=np.zeros(int(nh*(nh-1)/2))
    M2=np.zeros(nh*nh)
    M3=np.zeros(int(nh*(nh-1)/2))
    if(i<nh and j<nh):
        for l in range(na):
            for k in range(j):
                M1[int((l)*(l-1)/2+k)]=XL[i,k]*XL[j,l]-XL[i,l]*XL[j,k]
    if(i>nh and j>nh):
        for l in range(na,n):
            for k in range(na,j):
                M3[int((l-na)*(l-na-1)/2+k-na)]=XR[i-na,k-na]*XR[j-na,l-na]-XR[i-na,l-na]*XR[j-na,k-na]
    if(i<nh and j<nh):
        for l in range(na,n):
            for k in range(0,na):
                M2[int(l*na+k)]=XL[i,k]*XR[j,l]
    return M1, M2, M3
def second_order_compound_col_separated(XL,XR,l,k):
    n=len(XL)+len(XR)
    nh=na=nb=len(XL)
    M=np.zeros(int(n*(n-1)/2))
    M1=np.zeros(int(nh*(nh-1)/2))
    M2=np.zeros(nh*nh)
    M3=np.zeros(int(nh*(nh-1)/2))
    if(k<nh and l<nh):
        for j in range(na):
            for i in range(j):
                M1[int((j)*(j-1)/2+i)]=XL[i,k]*XL[j,l]-XL[i,l]*XL[j,k]
    if(k>nh and l>nh):
        for j in range(na,n):
            for i in range(na,j):
                M3[int((j-na)*(j-na-1)/2+i-na)]=XR[i-na,k-na]*XR[j-na,l-na]-XR[i-na,l-na]*XR[j-na,k-na]
    if(k<nh and l<nh):
        for j in range(na,n):
            for i in range(0,na):
                M2[int(j*na+i)]=XL[i,k]*XR[j,l]
    return M1, M2, M3
