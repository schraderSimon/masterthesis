import numpy as np

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
