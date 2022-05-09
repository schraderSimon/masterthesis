import numpy as np
import matplotlib.pyplot as plt
import autograd

def f(x,T):
    y=np.dot(x,T)
    return y**2

def jacobian(x,T):
    def intern_f(x):
        return f(x,T)
    jacob=autograd.jacobian(intern_f)
    return jacob

x=np.random.rand(10)
T=np.random.rand(10)
print(jacobian(x,T))
