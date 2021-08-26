import numpy as np
import matplotlib.pyplot as plt
def plot_eigenvectors(M,xmin,xmax,symmetric=False,num=5):
    n=1000
    if num>M(0).shape[0]:
        num=M(0).shape[0]
    xs=np.linspace(xmin,xmax,n)
    eigenvalues=np.zeros((num,n))
    for index,i in enumerate(xs):
        print(M(i))
        if symmetric:
            eigvals,eigvecs=np.linalg.eigh(M(i))
        else:
            eigvals,eigvecs=np.linalg.eig(M(i))
        eigenvalues[:,index]=eigvals[:num]
    for i in range(num):
        plt.plot(xs,eigenvalues[i,:])
    plt.show()
n=3
M=np.random.rand(n,n)*1
M=(M+M.T)/2
def generate_random_matrix_function(n,deg, M):
    numbers=np.random.rand(n,deg)
    def matrix(x):
        a=np.ones(n)
        for i in range(n):
            for j in range(deg):
                a[i]*=(x-numbers[i,j]) #random polynomial
        print(a)
        return M+np.diag(a)
    return matrix
matrix=generate_random_matrix_function(n,3,M)
plot_eigenvectors(matrix,-1,1,symmetric=True,num=2)
