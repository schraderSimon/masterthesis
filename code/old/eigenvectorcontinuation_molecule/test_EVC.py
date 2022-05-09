import numpy as np
from eigenvectorcontinuation import *
np.set_printoptions(linewidth=200,precision=4,suppress=True)
from scipy import linalg
X=np.random.rand(3,3)
XX=linalg.block_diag(X,X)
adj1=first_order_adj_matrix(XX)
adj2=second_order_adj_matrix(XX)
print(adj1)
print(adj2)
