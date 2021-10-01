import numpy as np
import matplotlib.pyplot as plt
from eigenvectorcontinuation import *
import scipy
np.set_printoptions(linewidth=200,precision=2,suppress=True)
X=np.arange(1.0,10.0).reshape(3,3)
X=np.ones((10,10))
X=np.random.rand(4,4)
Y=np.random.rand(4,4)

LDU_decomp(X)

adj_2_1,adj_2_2,adj_2_3=second_order_adj_matrix_blockdiag_separated(X,Y)
Y[0,0]=Y[0,0]+1
adj_2_1L,adj_2_2L,adj_2_3L=second_order_adj_matrix_blockdiag_separated(X,Y)
Y[0,0]=Y[0,0]-2
adj_2_1R,adj_2_2R,adj_2_3R=second_order_adj_matrix_blockdiag_separated(X,Y)
adj_2_1_NEW,adj_2_2_NEW,adj_2_3_NEW=0.5*(adj_2_1L+adj_2_1R),0.5*(adj_2_2L+adj_2_2R),0.5*(adj_2_3L+adj_2_3R)

print(adj_2_1)
print(adj_2_1_NEW)
print(adj_2_1-adj_2_1_NEW)
print(adj_2_2-adj_2_2_NEW)
print(adj_2_3-adj_2_3_NEW)
assert(np.all(np.abs(adj_2_1-adj_2_1_NEW)<1e-5))
assert(np.all(np.abs(adj_2_2-adj_2_2_NEW)<1e-5))
assert(np.all(np.abs(adj_2_3-adj_2_3_NEW)<1e-5))
