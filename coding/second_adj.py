import numpy as np
from scipy import linalg
from sympy import Matrix
Y=np.random.rand(10)+1
U=np.random.rand(10,10)
U=U+U.T
#eigval,eigvec= np.linalg.eig(X+X.T)
#X=eigvec@np.diag(Y)@eigvec.T
X=U@np.diag(Y)@U.T #random matrix with real eigenvalues
L,S,R=linalg.svd(X)
X=np.diag(S)
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
secondorder_adj=second_order_adj_matrix(X)
secondorder_comp=second_order_compound(X)
print(secondorder_adj)
print(secondorder_comp)
print(np.diag(secondorder_adj@secondorder_comp))
print(np.linalg.det(X))
