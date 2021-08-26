import numpy as np
import sys
import matplotlib.pyplot as plt
def eigenvec_cont(x,M,solv,ortho=False):
    """
    Implements the eigenvector continuation algorithm for the lowest eigenvalue.

    Returns: The estimated best eigenvector at point x

    Input:
        x: The point to solve the equation at
        M: The matrix function M(x)
        solv: The eigenvectors to use as a basis
        ortho: Wether the eigenvectors are diagonal (they are not, normally). - this still needs to be implemented!!
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
    s, U=np.linalg.eigh(S)
    s=np.diag(s)
    X=U@np.linalg.inv(np.sqrt(s))@U.T
    Tstrek=X.T@T@X
    epsilon, Cstrek = np.linalg.eigh(Tstrek)
    idx = epsilon.argsort()[::1]
    epsilon = epsilon[idx]
    Cstrek = Cstrek[:,idx]
    C=X@Cstrek

    lowest_eigenvalue=epsilon[0]
    return lowest_eigenvalue
def find_lowest_eigenvectors(matrix,xarr,symmetric=False):
    eigenvectors=np.zeros((len(xarr),matrix(0).shape[0])) #
    eigenvalues=np.zeros(len(xarr))
    for index,x in enumerate(xarr):
        if symmetric:
            eigvals,eigvecs=np.linalg.eigh(matrix(x))
        else:
            eigvals,eigvecs=np.linalg.eig(matrix(x))
        idx = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:, idx]
        sol=eigvecs[:,0]
        sol_val=eigvals[0]
        eigenvalues[index]=sol_val
        eigenvectors[index]=sol
    return eigenvectors, eigenvalues
n=50
M=np.random.rand(n,n)*1
M=(M+M.T)/2
#M=np.zeros((n,n))
def generate_random_matrix_function(n,deg, M):
    numbers=np.random.rand(n,deg)
    def matrix(x):
        a=np.ones(n)
        for i in range(n):
            for j in range(deg):
                a[i]*=(x-numbers[i,j]) #random polynomial
        return M+np.diag(a)
    return matrix
matrix=generate_random_matrix_function(n,3,M)
fig,(ax1,ax2)=plt.subplots(1,2,sharex=True)
x=[0]

xnew=np.linspace(-3,3,101)
eigvals=np.zeros(len(xnew))
true_eigenvectors,true_eigenvalues=find_lowest_eigenvectors(matrix,xnew,False)
dots=np.arange(0,10,1)
for i in range(1,6,1):
    x=dots[:i]
    eigenvectors,eigenvalues=find_lowest_eigenvectors(matrix,x,False)
    eigvalst,eigenvectt=np.linalg.eig(matrix(0))
    for j,xval in enumerate(xnew):
        eigvals[j]=eigenvec_cont(xval,matrix,eigenvectors,ortho=False)
    ax1.plot(xnew,eigvals,label="Continuation with i=%d"%i)
    ax2.plot(xnew,eigvals-true_eigenvalues,label="Continuation with i=%d"%i)
ax1.plot(xnew,true_eigenvalues,label="True")
ax2.plot(xnew,true_eigenvalues-true_eigenvalues,label="True")

ax1.legend()
ax2.legend()

plt.show()
