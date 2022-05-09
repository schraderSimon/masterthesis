import numpy as np
import sys
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200,precision=5,suppress=True)
def generalized_eigenvector(T,S,threshold=1e-8):
    """Solves the generalized eigenvector problem with canonical orthogonalization.
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
    s=s[s>threshold] #Keep only largest eigenvalues
    spowerminushalf=s**(-0.5) #Take s
    snew=np.zeros((len(U),len(spowerminushalf)))
    sold=np.diag(spowerminushalf)
    snew[:len(s),:]=sold
    s=snew

    ###Canonical orthogonalization
    X=U@s
    Tstrek=X.T@T@X
    epsilon, Cstrek = np.linalg.eigh(Tstrek)
    idx = epsilon.argsort()[::1] #Order by size (non-absolute)
    epsilon = epsilon[idx]
    Cstrek = Cstrek[:,idx]
    C=X@Cstrek
    lowest_eigenvalue=epsilon[0]
    lowest_eigenvector=C[:,0]
    return lowest_eigenvalue,lowest_eigenvector


def eigenvec_cont(x,M,solv,threshold=1e-15):
    """
    Implements the eigenvector continuation algorithm for the lowest eigenvalue.

    Input:
        x: The point to solve the equation at
        M: The matrix function M(x)
        solv: The eigenvectors to use as a basis
    Output:
        The eigenvector continuation guess of the eigenalue and the eigenvector at x
    """
    T=np.zeros((len(solv),len(solv)))  # M in the basis of the eigenvectors
    S=np.zeros((len(solv),len(solv)))  # The overlap matrix

    """Calculate products M(x)@v"""
    mv_prod=np.zeros_like(solv)
    for i in range(len(solv)):
        mv_prod[i]=M(x)@solv[i]

    """Calculate overlap matrix"""
    for i in range(len(solv)):
        for j in range(len(solv)):
            S[i,j]=solv[i].T@solv[j]
            T[i,j]=solv[i]@mv_prod[j]
    if len(solv)==5:

        s,U=np.linalg.eigh(S)
        print(S)
        for sval in s:
            print("%.4e"%sval)
        from numpyarray_to_latex import to_ltx

    return generalized_eigenvector(T,S,threshold=threshold)
def find_lowest_eigenvectors(matrix,xarr,num_levels=1,symmetric=False):
    """For each x in xarr, returns the lowest eigenvalue and eigenvector of a matrix function M(x)"""
    eigenvectors=np.zeros((len(xarr)*num_levels,matrix(0).shape[0])) #
    eigenvalues=np.zeros(len(xarr)*num_levels)
    for index,x in enumerate(xarr):
        if symmetric:
            eigvals,eigvecs=np.linalg.eigh(matrix(x))
        else:
            eigvals,eigvecs=np.linalg.eig(matrix(x))
        idx = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:, idx]

        sol=eigvecs[:,:num_levels]#,:num_levels]

        sol_val=eigvals[:num_levels]
        eigenvalues[num_levels*index:num_levels*(index+1)]=sol_val
        eigenvectors[num_levels*index:num_levels*(index+1),:]=sol.T

    return eigenvalues, eigenvectors
