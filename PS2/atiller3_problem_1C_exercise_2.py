# Problem Set 2, #1C

# Load tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp
from atiller3_problem_1A_exercise_2 import poiss_m
from atiller3_problem_1B_exercise_2 import *


def loglike(params, X, r, prior=None, sigma_p=1):
    """
    Generates log likelihood for given model for parameter estimation.
    Model:
    r = Xg + e , e Gaussian noise (assumed known as (~N(0,1)). r is dim (Mx1), X is dim (NxM), g is dim(Nx1)
                 the parameter sought, g is made up of a gaussian window * cosine function
    Log-likelihood to minimize:
    ll = || r - Xg || +  (1/σ²)|| p_g ||, where p_g is the prior of g
    :param params: Parameters to solve for
    :param X: ndarray, input into model
    :param y: ndarray, output of model
    :param prior: bool, inclusion of prior
    :param gauss_prior: bool, if True will generate Gaussian prior. If False, generates smoothed Poisson case prior.
    """
    g = params  # name parameters to be used
    n = len(g)

    # Establish prior
    if prior==None:
        log_prior = 0
        # log_prior = (1/(sigma_p)**2)*np.full((1,len(g)),1/n).dot(np.full((1,len(g)),1/n).T)
    elif prior=='gauss':
        log_prior = (1/(sigma_p)**2)*g.dot(g)
    elif prior=='poisson':
        log_prior = (1/(sigma_p)**2)*np.gradient(g).dot(np.gradient(g))

    # Create system to minimize
    log_sys = ((r-(X.T).dot(g)).dot((r-(X.T).dot(g)).T))
    return log_sys + log_prior


def mle_g(N,M,alpha=5,prior=None,A=1,sigma_p=1):
    its = 0;
    while np.min(its)<6: # For a minimum number of iterations before stopping
        rl, X, g = poiss_m(N=N, M=M, alpha=alpha, draws=1,A=A)
        r = X.T @ g + np.random.normal(0,1,1)
        bnds = [(-1,1)]*N # g on [-1,1], like cosine function
        params_0 = [np.full((1,N),0)] # initialize arbitrary initial guess
        gHat,_,its,_,_ = fmin_slsqp(loglike, params_0, f_eqcons=None,  args=(X,r,prior), bounds=bnds,
                                     iter=250, disp=False, full_output=True)
    return gHat,g,its,(X,g)

def plot_p1c(gHat, g, its, Nvec, Mvec, method):
    mse = [(np.square(gHat[i] - g[i]).sum() / len(g[i])) for i in range(len(g))] # calculate mse
    fig,ax = plt.subplots(4,3,figsize=[12,10]); axs = ax.ravel()
    sup=12;
    for i in np.arange(sup):
        axs[i].plot(gHat[i],'m',label='$\hat{g}$')
        axs[i].plot(g[i],'k',label='g')
        axs[i].set_xlabel('N'); axs[i].set_ylabel('Amplitude')
        axs[i].set_title(f"MSE = {mse[i]:.3f} for N={Nvec[i]}, M={Mvec[i]} over Iter: {its[i]}")
        axs[i].legend(loc='best', fontsize=8)
    fig.suptitle(r'$\hat{g}$ vs g for M=N, M=2N, M=$\frac{N}{2}$ pairs via '+method)
    fig.tight_layout(); plt.show()

# Plot MLE (no prior)
Nvec = [10, 10, 10, 50, 50, 50, 100, 100, 100, 200, 200, 200]
Mvec = [10, 20, 5,  50, 100,25, 100, 200, 50,  200, 400, 100]
prior = None
gHat,g,its,_ = map(list, zip(*[mle_g(N=Nvec[i], M=Mvec[i], prior=prior) for i in range(len(Nvec))]))
gHat = np.array(gHat,dtype='object'); g = np.array(g,dtype='object'); np.array(its)
plot_p1c(gHat,g, its, Nvec, Mvec, method=f'MLE, prior: {prior}')