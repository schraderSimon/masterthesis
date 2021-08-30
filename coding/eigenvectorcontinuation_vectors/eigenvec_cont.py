import numpy as np
import sys
import matplotlib.pyplot as plt
def eigenvec_cont(x,M,solv):
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

    """Perform Løwdin orthogonalization of the basis"""
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
    lowest_eigenvector=C[:,0]
    return lowest_eigenvalue,lowest_eigenvector
def find_lowest_eigenvectors(matrix,xarr,num_levels=1,symmetric=False):
    """For each x in xarr, returns the lowest eigenvalue and eigenvector of a matrix function M(x)"""
    eigenvectors=np.zeros((len(xarr)*num_levels,matrix(0).shape[0])) #
    eigenvalues=np.zeros(len(xarr)*num_levels)
    print(num_levels)
    for index,x in enumerate(xarr):
        if symmetric:
            eigvals,eigvecs=np.linalg.eigh(matrix(x))
        else:
            eigvals,eigvecs=np.linalg.eig(matrix(x))
        idx = eigvals.argsort()
        eigvals.sort()
        eigvecs = eigvecs[:, idx]
        #sol=eigvecs[:,0]
        #sol_val=eigvals[0]
        #print(eigenvalues)
        #eigenvalues[index]=sol_val
        #eigenvectors[index]=sol
        sol=eigvecs[:,:num_levels]#,:num_levels]

        sol_val=eigvals[:num_levels]
        print(sol_val)
        eigenvalues[num_levels*index:num_levels*(index+1)]=sol_val
        print(eigenvectors.shape)
        print(sol.shape)
        print(eigenvectors[2:4,:].shape)
        eigenvectors[num_levels*index:num_levels*(index+1),:]=sol.T

    return eigenvalues, eigenvectors
def generate_random_matrix_function(n,deg, M,a=0,b=1,scaling=1):
    """Generates a random Hermitian matrix function.
    Generates a random matrix function. Hermiticity is remained
    by influencing the diagonal elements only.
    The diagonal elements are each modelled as a monic polynomial of degree "deg",
    with zeros in [a,b).
    Off diagonal matrix elements take values [0,1). The matrix can be scaled by modifying the value "scaling".
    Input:
        n - the dimension of the matrix
        deg - Degree of the random polynomial
        a, b - range of the zeros of the polynomial
        scaling - factor to multiply the matrix with
    Returns:
        Random matrix function M(x)
    """
    numbers=(b-a)*np.random.rand(n,deg)+a
    def matrix(x):
        A=np.ones(n)
        for i in range(n):
            for j in range(deg):
                A[i]*=(x-numbers[i,j]) #random polynomial
        return (M+np.diag(A))*scaling
    return matrix


if __name__=="__main__":
    num_samples=10
    n=100
    M=np.random.rand(n,n)*1
    M=(M+M.T)/2
    matrix=generate_random_matrix_function(n,3,M,a=0,b=2)
    fig,(ax1,ax2)=plt.subplots(1,2,sharex=True)
    x=[0]

    xnew=np.linspace(-1,1,1001)
    eigvals=np.zeros(len(xnew))
    true_eigenvalues,true_eigenvectors=find_lowest_eigenvectors(matrix,xnew,False)
    dots=np.arange(0,100,0.1)
    for i in range(1,6,1):
        x=dots[:i]
        eigenvalues,eigenvectors=find_lowest_eigenvectors(matrix,x,False)
        eigvalst,eigenvectt=np.linalg.eig(matrix(0))
        for j,xval in enumerate(xnew):
            eigvals[j],soppel=eigenvec_cont(xval,matrix,eigenvectors)
        ax1.plot(xnew,eigvals,label="Continuation with i=%d"%i)
        ax2.plot(xnew,eigvals-true_eigenvalues,label="Continuation with i=%d"%i)
    ax1.plot(xnew,true_eigenvalues,label="True")
    ax2.plot(xnew,true_eigenvalues-true_eigenvalues,label="True")
    ax1.set_title("Eigenvalues")
    ax2.set_title("Deviation from true eigenvalues")
    ax1.legend()
    ax2.legend()

    plt.show()
