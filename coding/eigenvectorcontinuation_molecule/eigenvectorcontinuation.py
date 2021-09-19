import numpy as np
from scipy import linalg
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
def first_order_adj_matrix(X,detX=None):
    if detX is None:
        detX=np.linalg.det(X)
    return detX*linalg.inv(X)
def second_order_compound(X):
    n=len(X)
    M=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)))
    for j in range(n):
        for i in range(j):
            for l in range(n):
                for k in range(l):
                    M[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=X[i,k]*X[j,l]-X[i,l]*X[j,k]
    return M

def second_order_adj_matrix(X,detX=None):
    if detX is None:
        detX=np.linalg.det(X)
    first_order=first_order_adj_matrix(X,detX)
    return 1/detX*second_order_compound(first_order)
