from matrix_operations import *
import numpy as np
np.set_printoptions(linewidth=200,precision=5,suppress=True)

X=np.random.rand(5,3)
D=2*X@X.T
R,P=cholesky_pivoting(2*X@X.T)
PL=P@R.T
Cnew=PL[:,:X.shape[1]]/np.sqrt(2)
print((D))
print(PL@PL.T)
print(2*Cnew@Cnew.T)
assert np.all((X@X.T-Cnew@Cnew.T)<1e-10)
