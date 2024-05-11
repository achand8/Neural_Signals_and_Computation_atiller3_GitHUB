# Problem Set 2, #2A

# Load tools
import numpy as np
import scipy.io as sio
import scipy
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


def gp_smoothing(A, l, x):
    '''
    :param A: float, scalar hyperparameter of GP estimate
    :param l: float, length parameter, hyperparameter of GP estimate
    :param x: 1-d numpy array, data whose timepoints are used to fit to GP
    :return gp_smooth: 1-d array, GP fit for x
    '''
    t=np.expand_dims(np.linspace(0, len(x), 400), 1)/1000
    d = scipy.spatial.distance.cdist(t, t, 'sqeuclidean')
    K = A * np.exp(-d/l)     # compute kernel
    invK = np.linalg.pinv(K)

    # Fit the GP for predictive distribution such that
    # mu = KT Kinv x
    # cov = K - KT K-1 K
    mu = K.T.dot(invK).dot(x)
    cov = K - K.T.dot(invK).dot(K)
    gp_smooth=np.random.multivariate_normal(mean=
                                            mu,
                                            cov=cov,size=1)
    # gp_smooth[gp_smooth<0] = 0  # limit gp to whole numbers
    return gp_smooth.T