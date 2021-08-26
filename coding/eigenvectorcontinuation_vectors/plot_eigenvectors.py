import numpy as np
import matplotlib.pyplot as plt
from eigenvec_cont import *
def plot_eigenvectors(M,x,symmetric=False,num=5):
    """Plots, for each x, the 'num' first eigenvalues of M(x)"""
    n=len(x)
    if num>M(0).shape[0]:
        num=M(0).shape[0]
    eigenvalues=np.zeros((num,n))
    for index,i in enumerate(x):
        print(M(i))
        if symmetric:
            eigvals,eigvecs=np.linalg.eigh(M(i))
        else:
            eigvals,eigvecs=np.linalg.eig(M(i))
        eigenvalues[:,index]=eigvals[:num]
    for i in range(num):
        plt.plot(x,eigenvalues[i,:])
    plt.show()

if __name__=="__main__":
    n=10
    M=np.random.rand(n,n)*1
    M=(M+M.T)/2
    matrix=generate_random_matrix_function(n,3,M)
    x=np.linspace(-1,1,100)
    plot_eigenvectors(matrix,x,symmetric=True,num=10)
