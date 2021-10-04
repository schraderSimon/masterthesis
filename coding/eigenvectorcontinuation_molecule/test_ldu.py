import numpy as np
import matplotlib.pyplot as plt
from eigenvectorcontinuation import *
import scipy
import sys
np.set_printoptions(linewidth=200,precision=2,suppress=True)
X=np.arange(1.0,10.0).reshape(3,3)
X=np.ones((10,10))
plussy=0
minussy=0
for i in range(1000):
    try:
        X=np.random.rand(4,4)@np.diag(np.concatenate((np.random.rand(2)+0.3,np.array([1,1]))))@np.random.rand(4,4)*2
        Y=np.random.rand(4,4)@np.diag(np.concatenate((np.random.rand(2)+0.3,np.random.rand(2)+0.3)))@np.random.rand(4,4)+0.5*2
        #Linv,d,Rinv,P=LDU_decomp(X)
        #print(X[np.ix_(np.arange(len(X)),P)])
        #print(Linv@np.diag(d)@Rinv)
        S=scipy.linalg.block_diag(X,Y)
        L_inv,d,R_inv=LDU_decomp_newnew(S)
        assert (np.all(np.abs(L_inv@np.diag(d)@R_inv-S)<1e-5))
        print(np.linalg.det(S))
        L=np.linalg.inv(L_inv)
        R=np.linalg.inv(R_inv)
        d_2=second_order_adj_matrix(np.diag(d))
        R2=second_order_compound(R)
        L2=second_order_compound(L)
        S_2=second_order_adj_matrix(S)
        print(d_2)
        #shouldbeS2=R2@d_2@L2
        shouldbeS2=second_order_adj_matrix(R_inv)@d_2@second_order_adj_matrix(L_inv)
        assert(np.all(abs(second_order_adj_matrix(R_inv)-R2)<1e-5))

        assert(np.all(abs(second_order_adj_matrix(L_inv)-L2)<1e-5))
        #assert(np.max(np.abs(shouldbeS2))>1e-3)
        #print(np.argwhere(np.abs(shouldbeS2)>1))
        #print(shouldbeS2)
        #print(S_2)
        assert((np.all(abs(shouldbeS2+S_2)<1e-5) or np.all(abs(shouldbeS2-S_2)<1e-5)))
        if(np.all(abs(shouldbeS2+S_2)<1e-5)):
            plussy+=1
        if(np.all(abs(shouldbeS2-S_2)<1e-5)):
            minussy+=1
    except np.linalg.LinAlgError:
        continue
    print(minussy,plussy)
sys.exit(1)










for i in range(1000):
    try:
        X=np.random.rand(4,4)@np.diag(np.concatenate((np.random.rand(2)+0.1,np.array([1e-1,1e-1]))))@np.random.rand(4,4)*2
        Y=np.random.rand(4,4)@np.diag(np.concatenate((np.random.rand(2),np.random.rand(2))))@np.random.rand(4,4)+0.5*2
        #Linv,d,Rinv,P=LDU_decomp(X)
        #print(X[np.ix_(np.arange(len(X)),P)])
        #print(Linv@np.diag(d)@Rinv)
        S=scipy.linalg.block_diag(X,Y)
        S=X
        L,d,U,P=LDU_decomp_new(S)





        L_inv,d,R_inv,P=LDU_decomp(S)
        R_inv=R_inv[np.ix_(np.arange(len(S)),np.argsort(P))]
        R_inv[0,:]=R_inv[0,:]*parity(P)
        d[0]=d[0]*parity(P)
        print(np)
        assert (np.all(np.abs(L_inv@np.diag(d)@R_inv-S)<1e-5))
        print(np.linalg.det(S))
        L=np.linalg.inv(L_inv)
        R=np.linalg.inv(R_inv)
        d_2=second_order_adj_matrix(np.diag(d))
        R2=second_order_compound(R)
        L2=second_order_compound(L)
        S_2=second_order_adj_matrix(S)
        print(d_2)
        #shouldbeS2=R2@d_2@L2
        shouldbeS2=second_order_adj_matrix(R_inv)@d_2@second_order_adj_matrix(L_inv)
        assert(np.all(abs(second_order_adj_matrix(R_inv)-R2)<1e-5))

        assert(np.all(abs(second_order_adj_matrix(L_inv)+L2)<1e-5))
        #assert(np.max(np.abs(shouldbeS2))>1e-3)
        #print(np.argwhere(np.abs(shouldbeS2)>1))
        #print(shouldbeS2)
        #print(S_2)
        assert((np.all(abs(shouldbeS2+S_2)<1e-5) or np.all(abs(shouldbeS2-S_2)<1e-5)))
        if(np.all(abs(shouldbeS2+S_2)<1e-5)):
            plussy+=1
        if(np.all(abs(shouldbeS2-S_2)<1e-5)):
            minussy+=1
            par=parity(P)
            print("Parity: %d"%(par))
            print("S")
            print(S)
            detL=np.linalg.det(L)
            detR=np.linalg.det(R)
            detS=np.linalg.det(S)
            detD=np.linalg.det(np.diag(d))
            print("Determinants of L (%.2f) and R (%.2f), D (%e) and S (%e)"%(detL,detR,detD,detS))
            print("Signs: %d, %d"%(np.sign(detS),np.sign(detL*detR*detD*par)))
    except np.linalg.LinAlgError:
        continue
    print(minussy,plussy)
