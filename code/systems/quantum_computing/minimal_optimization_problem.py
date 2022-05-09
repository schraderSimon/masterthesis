import numpy as np
import scipy
"""
def fun(x,E,cstrek):
    c0=x[0]
    c1=x[1]
    c2=x[2]
    lambda1=x[3]
    lambda2=x[4]
    E0=E[0]
    E1=E[1]
    E2=E[2]
    c0s=cstrek[0]
    c1s=cstrek[1]
    c2s=cstrek[2]
    eq1=2*c0*(E0+lambda1)+lambda2*c0s
    eq2=2*c1*(E1+lambda1)+lambda2*c1s
    eq3=2*c2*(E2+lambda1)+lambda2*c2s
    eq4=c0**2+c1**2+c2**2-1
    eq5=c0*c0s+c1*c1s+c2*c2s-0
    return [eq1,eq2,eq3,eq4,eq5]

from scipy import optimize
E=[1,2,3]
cstrek=np.array([np.sqrt(0.989),np.sqrt(0.01),np.sqrt(0.001)])
print(cstrek.T@cstrek)
sol=optimize.root(fun,np.concatenate(([0,1,0],[0,0])),args=(E,cstrek),method="hybr",tol=1e-15)
relsol=sol.x[:3]
print(relsol)
H=np.diag(E)
Hspace=np.zeros((2,2))
Hspace[0,0]=cstrek.T@H@cstrek
Hspace[0,1]=Hspace[1,0]=cstrek.T@H@relsol
Hspace[1,1]=relsol.T@H@relsol
print(Hspace)
print(np.linalg.eigh(Hspace)[0])
"""

M=np.random.rand(4,4)
P=np.random.rand(4,4)
M=M+M.T
P=P+P.T
print("Ms lowest eigenvalue")
print(np.linalg.eigh(M)[1][0])
print("Ms lowest eigenvalue post perturbation")
print(np.linalg.eigh(M+0.001*P)[1][0])
print("Difference:")
print(np.linalg.eigh(M+0.001*P)[1][0]-np.linalg.eigh(M)[1][0])
print("Ps lowest eigenvalue:")
print(np.linalg.eigh(P)[1][0])
