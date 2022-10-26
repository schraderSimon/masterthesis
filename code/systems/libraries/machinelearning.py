import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import sys
from scipy.linalg import block_diag
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, PairwiseKernel, DotProduct
from scipy.optimize import minimize, minimize_scalar

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
def RBF_kernel_unitary_matrices(list_U1,list_U2,kernel_params=[1,1,1,1,1]):
    if kernel_params[1]<-0.5:
        kernel_params[1]=-0.5
    l_1=np.exp(kernel_params[1])
    sigma_1=np.exp(kernel_params[0])
    noise=0#np.exp(kernel_params[4])
    norm=np.zeros((len(list_U1),len(list_U2)))
    n=min([len(list_U1),len(list_U2)])
    for i in range(len(list_U1)):
        for j in range(len(list_U2)):
            U1=list_U1[i]
            U2=list_U2[j]
            norm[i,j]=np.linalg.norm(U1-U2)
    kernel_mat= sigma_1*np.exp(-0.5*norm**2/l_1)
    for i in range(n):
        kernel_mat[i,i]+=noise
    return kernel_mat
def extended_RBF_kernel_unitary_matrices(list_U1,list_U2,kernel_params=[1,1,1,1,1]):
    sigma_1=np.exp(kernel_params[0])
    l_1=np.exp(kernel_params[1])
    sigma_2=np.exp(kernel_params[2])
    l_2=np.exp(kernel_params[3])
    sigma_3=np.exp(kernel_params[4])
    noise=0
    norm=np.zeros((len(list_U1),len(list_U2)))
    n=min([len(list_U1),len(list_U2)])
    for i in range(len(list_U1)):
        for j in range(len(list_U2)):
            U1=list_U1[i]
            U2=list_U2[j]
            norm[i,j]=np.linalg.norm(U1-U2)
    #kernel_mat= sigma*np.exp(-0.5*norm**2/l)
    kernel_mat= sigma_1*np.exp(-0.5*norm**2/l_1)+sigma_2*np.exp(-0.5*norm/l_2)
    for i in range(n):
        kernel_mat[i,i]+=noise
    return kernel_mat

def RQ_kernel_unitary_matrices(list_U1,list_U2,kernel_params=[1,1,100]):
    sigma=np.exp(kernel_params[0])
    l=np.exp(kernel_params[1])
    alpha=np.exp(kernel_params[2])
    norm=np.zeros((len(list_U1),len(list_U2)))
    for i in range(len(list_U1)):
        for j in range(len(list_U2)):
            U1=list_U1[i]/np.mean(list_U1[i]**2)
            U2=list_U2[j]/np.mean(list_U2[j]**2)
            norm[i,j]=np.linalg.norm(U1-U2)
    kernel_mat=sigma**2*(np.ones_like(norm)+0.5*norm**2/(alpha*l**2))**(-alpha)
    return kernel_mat

def GP(X1, y1, X2, kernel_func,kernel_params):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1,kernel_params)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2,kernel_params)
    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2,kernel_params)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance
def log_likelihood(kernel_params,data_X,y,kernel):
    cov_matrix=kernel(data_X,data_X,kernel_params) #The covariance matrix
    cov_matrix=cov_matrix+np.eye(len(cov_matrix))*1e-14
    det=np.linalg.det(cov_matrix)
    log_det=np.log(det)
    #inverse=np.linalg.inv(cov_matrix+1e-10*np.eye(len(cov_matrix)))
    inv_times_data=np.linalg.solve(cov_matrix,y)
    return -(-0.5*y.T@inv_times_data-0.5*log_det)
def find_best_model(U_list,y,kernel,start_params):
    y_new=y
    sol=minimize(log_likelihood,x0=start_params,args=(U_list,y,kernel),bounds=[(None,None),(0.5,None),(None,None),(1,None),(None,None)])
    best_sigma=sol.x
    print(best_sigma,sol.fun)
    #print(kernel(U_list,U_list,sol.x))
    return best_sigma
def get_model(U_list,y,kernel,U_list_target,start_params=[-2,0,0,0,0]):
    best_sigma=find_best_model(U_list,y,kernel,start_params)
    new, Σ2 = GP(U_list, y, U_list_target, kernel,best_sigma)
    return new, np.diag(Σ2)
