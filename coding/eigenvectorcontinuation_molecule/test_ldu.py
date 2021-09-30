import numpy as np
import matplotlib.pyplot as plt
from eigenvectorcontinuation import *
np.set_printoptions(linewidth=200,precision=2,suppress=True)
X=np.arange(1.0,10.0).reshape(3,3)
X=np.ones((10,10))
#X=np.random.rand(5,5)

LDU_decomp(X)
