# Problem Set 2, #2A

# Load tools
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pkg_resources
import pip


# Load the N x T data
ndat = sio.loadmat('sample_dat.mat')['dat']
dfn = pd.DataFrame(ndat[0])
spk_arr = np.array(dfn['spikes'][:])
psth = spk_arr.sum(axis=0)

# Plot the PSTH in subplots
def plot_2a(y,title):
    plt.rc(('xtick','ytick'),labelsize=10)
    fig, ax = plt.subplots(3,2, figsize=(12,8)); axs=ax.ravel()
    [(axs[i].bar(np.arange(len(y.T)),y[i,:].T),
      axs[i].set_title(f'Neuron {i+1}',fontsize=10))
     for i in np.arange(6)]
    matplotlib.rcParams.update({'font.size':10})
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def plot_2a_gp(y, title):
    plt.rc(('xtick', 'ytick'), labelsize=10)
    fig, ax = plt.subplots(3, 2, figsize=(12, 8));
    axs = ax.ravel()
    [(axs[i].plot(np.arange(len(y.T)), y[i, :].T),
      axs[i].set_title(f'Neuron {i + 1}', fontsize=10))
     for i in np.arange(6)]
    matplotlib.rcParams.update({'font.size': 10})
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


# Perform GP smoothing
def gp_smoothing(A, l, x):
    '''
    :param A: float, scalar hyperparameter of GP estimate
    :param l: float, length parameter, hyperparameter of GP estimate
    :param x: 1-d numpy array, data to fit to GP
    :return gp_smoooth: 1-d array, GP fit for x
    '''
    t = x.astype('int64')
    d = t-t[:,np.newaxis]
    K = A * np.exp(-np.square(d)/l)
    invK = np.linalg.pinv(K)

    # Fit the GP for predictive distribution such that
    # mu = KT Kinv x
    # cov = K - KT K-1 K
    mu = K.T.dot(invK).dot(t)
    cov = K - K.T.dot(invK).dot(K)
    gp_smooth=np.random.multivariate_normal(mean=
                                            mu.ravel(),
                                            cov=cov,size=5)
    gp_smooth=gp_smooth.mean(axis=0)
    return gp_smooth.T


# Plot the PSTH and some GP-smoothed PSTHs
# plot_2a(psth,'PSTH'); A=2; l=0.2
# gp_smooth = np.array([gp_smoothing(A=A, l=l, x=psth[i,:]) for i in np.arange(psth.shape[0])]).reshape(psth.shape)
# plot_2a(gp_smooth,f'PSTH - GP smoothing at A={A}, l={l}')
# # Try far-out parameters for A and l
# A=2; l=20; gp_smooth = np.array([gp_smoothing(A=A, l=l, x=psth[i,:]) for i in np.arange(psth.shape[0])]).reshape(psth.shape)
# plot_2a(gp_smooth,f'PSTH - GP smoothing at A={A}, l={l}')

