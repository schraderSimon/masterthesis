import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import sys
from scipy.linalg import block_diag
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, PairwiseKernel, DotProduct
def multivariate_gaussian_scikitlearn(x_learn,y_learn,x_predict,sigma=1,l=1):
    kernel = RBF(length_scale=l,length_scale_bounds=(0.5, 10.0))#+np.mean(y_learn)*ConstantKernel(constant_value=1.0)
    gaussian_process = GaussianProcessRegressor(kernel=kernel)
    gaussian_process.fit(x_learn.reshape(-1, 1), y_learn.reshape(-1, 1))
    print(gaussian_process.get_params())
    mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(x_predict.reshape(-1, 1),return_std=True)
    return mean_predictions_gpr.ravel(), std_predictions_gpr.ravel()
def multivariate_gaussian_gpy(x_learn,y_learn,x_predict,sigma=1,l=1):
    i=1 #Setting i=0 is a bad idea - we should not scale the data!
    x_learn=np.array(x_learn)
    x_predict=np.array(x_predict)
    if i==0:
        x_learn_mean=np.mean(x_learn,axis=0)
        x_learn_std=np.std(x_learn,axis=0)
        x_learn=(x_learn-x_learn_mean)/x_learn_std
        x_predict=(x_predict-x_learn_mean)/x_learn_std
    if len(x_learn.shape)==1:
        x_learn=x_learn.reshape(-1,1)
        x_predict=x_predict.reshape(-1,1)
    m = GPy.models.GPRegression(x_learn,y_learn.reshape(-1, 1),kernel = GPy.kern.RBF(input_dim=x_learn.shape[1]))
    m["Gaussian_noise.variance"]=0
    m.optimize("bfgs")
    mean,var=m.predict(x_predict)
    return mean.ravel(),np.sqrt(var.ravel())
def multivariate_gaussian_gpy_matrixInput(x_learn,y_learn,x_predict,sigma=1,l=1):
    i=1#Setting i=0 is a bad idea - we should not scale the data!
    x_learn_new=[]
    for x in x_learn:
        x_learn_new.append(x.flatten())
    x_learn=x_learn_new
    x_predict_new=[]
    for x in x_predict:
        x_predict_new.append(x.flatten())
    x_predict=x_predict_new
    x_learn=np.array(x_learn)

    x_predict=np.array(x_predict)
    if i==0:
        x_learn_mean=np.mean(x_learn,axis=0)
        x_learn_std=np.std(x_learn,axis=0)
        x_learn=(x_learn-x_learn_mean)/x_learn_std
        x_predict=(x_predict-x_learn_mean)/x_learn_std
    m = GPy.models.GPRegression(x_learn,y_learn.reshape(-1, 1),kernel = GPy.kern.RBF(input_dim=x_learn.shape[1]))
    m["Gaussian_noise.variance"]=1e-16
    m.optimize("bfgs")
    mean,var=m.predict(x_predict)
    return mean.ravel(),np.sqrt(var.ravel())
