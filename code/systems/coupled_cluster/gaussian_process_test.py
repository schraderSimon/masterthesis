import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
from scipy.linalg import block_diag
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, PairwiseKernel, DotProduct, RationalQuadratic, Matern
import sys
from scipy.optimize import minimize
np.set_printoptions(precision=4,linewidth=400)
def multivariate_gaussian_scikitlearn(x_learn,y_learn,x_predict,sigma=1,l=1):
    mean_y=np.mean(y_learn)
    y_learn=y_learn-mean_y
    print(mean_y)
    kernel = sigma**2* RBF(length_scale=l, length_scale_bounds=(1, 10))
    #kernel= RationalQuadratic(length_scale=1.0, alpha=0.1,length_scale_bounds=(0.1, 10), alpha_bounds=(1e-2, 1e2))
    #kernel=sigma**2*Matern(length_scale=l, length_scale_bounds=(1, 10))
    gaussian_process = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9,normalize_y=False)
    gaussian_process.fit(x_learn.reshape(-1, 1), y_learn.reshape(-1, 1))
    mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(x_predict.reshape(-1, 1),return_std=True)
    return mean_predictions_gpr.ravel()+mean_y, std_predictions_gpr.ravel()

A=[[0.10418752,0.08528442,0.06861378,0.05533604,0.04472602,0.03590435,0.02844118,0.02229525,0.01746732,0.01381569],
[0.08528442,0.08950578,0.08241256,0.07081346,0.05932171,0.04958711,0.04180712,0.03575377,0.03113806,0.02769139],
[0.06861378,0.08241256,0.08811346,0.08495789,0.07743484,0.06949881,0.06282551,0.05766118,0.05378608,0.05092371],
[0.05533604,0.07081346,0.08495789,0.0933877,0.09545134,0.09383511,0.0912453,0.08899126,0.08741258,0.08646773],
[0.04472602,0.05932171,0.07743484,0.09545134,0.10974635,0.1192375,0.1253097,0.12968278,0.13327967,0.13642496],
[0.03590435,0.04958711,0.06949881,0.09383511,0.1192375,0.14255029,0.1621197,0.17794029,0.19063577,0.20074636],
[0.02844118,0.04180712,0.06282551,0.0912453,0.1253097,0.1621197,0.19800916,0.22969672,0.25561188,0.27579607],
[0.02229525,0.03575377,0.05766118,0.08899126,0.12968278,0.17794029,0.22969672,0.27912772,0.3213433,0.35443489],
[0.01746732,0.03113806,0.05378608,0.08741258,0.13327967,0.19063577,0.25561188,0.3213433,0.38062041,0.42895622],
[0.01381569,0.02769139,0.05092371,0.08646773,0.13642496,0.20074636,0.27579607,0.35443489,0.42895622,0.4942041]]
A=np.array(A)
sample_x=np.linspace(1.5,5,len(A))
x_of_interest=np.linspace(1.5,8,100)
num=1
"""
for i in range(3,len(A),6):
    input=sample_x[::num]
    output=A[i][::num]
    output_mean=np.mean(output)
    mean,std=multivariate_gaussian_scikitlearn(sample_x[::num],A[i][::num],x_of_interest,sigma=10,l=0.5)
    plt.plot(sample_x,A[i],label="A[%d]"%i)
    plt.plot(x_of_interest,mean,label="mean %d"%i)
    plt.fill_between(x_of_interest,mean+1.96*std,mean-1.96*std,color="grey",alpha=0.3)
plt.legend()
plt.show()
"""
"""
Gaussian Processes regression examples
"""
MPL_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    MPL_AVAILABLE = False

import numpy as np
import GPy

def toy_rbf_1d(optimize=True, plot=True):
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    try:
        import pods
    except ImportError:
        print("pods unavailable, see https://github.com/sods/ods for example datasets")
        return
    data = pods.datasets.toy_rbf_1d()

    # create simple GP Model
    input=sample_x[::num]
    output=A[2][::num]
    m = GPy.models.GPRegression(input.reshape(-1, 1),output.reshape(-1, 1),kernel = GPy.kern.RBF(1),noise_var=1e-5)
    m["Gaussian_noise.variance"]=0
    #help(m)
    if optimize:
        m.optimize("bfgs")
    mean,var=m.predict(input.reshape(-1,1))
    print(mean-output.reshape(-1, 1))
    print(var)
    if MPL_AVAILABLE and plot:
        m.plot()
    plt.show()
    return m
toy_rbf_1d()
