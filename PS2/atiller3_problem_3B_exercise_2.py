# Problem Set 2, #3B


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize

edat = sio.loadmat('exampleData.mat')['Data'][0]
reach27 = edat[26][0].T


def minA(x,dt=10e-3):
    """
    Generates A via least squares solution of minimization over A:
       argminL_A  = || ( xt - x(t-1) ) / dt - Ax(t-1) || +  (1/σ²)|| prior ||, and there is no prior here
    :param x: ndarray of dims neurons x time
    :return A: ndarray of dims neuron x neuron. 
    """

    xt = x[:,1:]; xt_1 = x[:,0:-1] + np.random.normal(0,1,xt.shape)
    y = (xt-xt_1)/dt # dt is bin width, or 10ms
    Xpinv = xt_1.T @ np.linalg.pinv(xt_1 @ xt_1.T)
    A = y @ Xpinv
    return A


def recon(A,x,dt=10e-3):
    """
    Reconstruct x at future time points using the current time point and a state-transition matrix using
        (xt+1 - xt)/dt = A xt ==> xt+1 = (A xt)dt + xt under assumed noise
    :param A: ndarray of dims neuron x neuron
    :param x: ndarray of dims neurons x time
    :return xt1, mse: xt1 is the reconstructed x at the next time point, and mse are reconstruction errors
    """
    xt = x[:,0:-1]
    xt1 = (A @ xt + np.random.normal(0,1,xt.shape)*dt + xt
    mse = np.linalg.norm(xt1 - x[:,1:],axis=0)
    plt.bar(np.arange(mse.shape[0]), mse); plt.xlabel('Time (bins)'); plt.ylabel('Error, norm')
    plt.title('Error of reconstruction, norm')
    plt.tight_layout()
    plt.show()
    return xt1, mse


def plot_recons(xt1, x):
    """
    Plot some reconstructions of neuronal activity using the computed A.
    :param xt1: ndarray, dims neurons x time - 1. Estimated x at next time point
    :param x: ndarray, dim neurons x time. Neuronal data.
    :return:
    """
    fig,ax = plt.subplots(3,3,figsize=[12,8])
    axs = ax.ravel(); xt = x[:,1:]
    domain = np.arange(xt1.shape[1])
    [(axs[i].plot(domain, xt1[i,:], 'm') , axs[i].plot(domain, xt[i,:], 'k'),
      axs[i].set_title(f'Neuron {i+1}'), (axs[i].set_xlabel('Time (bins)'),
                                          (axs[i].set_ylabel('Activity'))))
     for i in np.arange(9)]
    fig.tight_layout()
    plt.show()

x=reach27
A = minA(x=x)
xt1,mse = recon(A,x)
plot_recons(xt1, x)
plt.show()