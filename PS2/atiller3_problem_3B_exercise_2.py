# Problem Set 2, #3B


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize

edat = sio.loadmat('exampleData.mat')['Data'][0]
reach27 = edat[26][0].T


def minA(x, dt=10e-3):
    """
    Generates A via least squares solution of minimization over A:
       argminL_A  = || ( xt - x(t-1) ) / dt - Ax(t-1) || +  (1/σ²)|| prior ||, and there is no prior here
    :param x: ndarray of dims neurons x time
    :return A: ndarray of dims neuron x neuron.
    """

    xt = x[:, 1:];
    xt_1 = x[:, 0:-1]
    y = (xt - xt_1) / dt + np.random.normal(0, .4, xt.shape)  # dt is bin width, or 10ms
    Xpinv = xt_1.T @ np.linalg.pinv(xt_1 @ xt_1.T)
    A = y @ Xpinv
    return A


def recon(A, x, dt=10e-3):
    """
    Reconstruct x at future time points using the current time point and a state-transition matrix using
        (xt+1 - xt)/dt = A xt ==> xt+1 = (A xt)dt + xt under assumed noise
    :param A: ndarray of dims neuron x neuron
    :param x: ndarray of dims neurons x time
    :return xt1, err: xt1 is the reconstructed x at the next time point, and err are reconstruction errors
    """
    xt = x[:, 0:-1]
    xt1 = (A @ xt) * dt + xt;
    xt1 = np.concatenate((np.array(x[:, 0])[:, np.newaxis], xt1), axis=1)
    err = np.linalg.norm(xt1 - x, axis=0)
    plt.bar(np.arange(err.shape[0]), err);
    plt.xlabel('Time (bins)');
    plt.ylabel('Error, norm')
    plt.title('Error of reconstruction, norm')
    plt.tight_layout()
    plt.show()
    return xt1, err


def plot_recons(xt1, x, method):
    """
    Plot some reconstructions of neuronal activity using the computed A.
    :param xt1: ndarray, dims neurons x time - 1. Estimated x at next time point
    :param x: ndarray, dim neurons x time. Neuronal data.
    :param method: string, method appended to title for the figure
    :return:
    """
    fig, ax = plt.subplots(3, 3, figsize=[12, 8])
    axs = ax.ravel();
    domain = np.arange(xt1.shape[1])
    [(axs[i].plot(domain, x[i, :], 'k', label='orig'), axs[i].plot(domain, xt1[i, :], 'm', label='estim'),
      axs[i].set_title(f'Neuron {i + 1}'), (axs[i].set_xlabel('Time (bins)'),
                                            (axs[i].set_ylabel('Activity'))), axs[i].legend())
     for i in np.arange(9)]

    fig.suptitle(f'Reconstruction via {method}')
    fig.tight_layout()
    plt.show()