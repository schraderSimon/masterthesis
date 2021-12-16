import numpy as np
import matplotlib.pyplot as plt
import autograd
T=np.random.rand(10)
def f(x):
    return x**2-x+2-np.dot(x,x)
def jacobian():
    return autograd.jacobian(f)

x=np.random.rand(10)
print(autograd.jacobian(f)(x))
xs=x.copy()
xs[1]+=1e-8
print((f(xs)-f(x))/1e-8)
