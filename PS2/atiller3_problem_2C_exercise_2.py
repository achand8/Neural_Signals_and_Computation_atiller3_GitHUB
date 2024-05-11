# Problem Set 2, #2C

# Load tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pkg_resources
import scipy.io as sio
import pip

# Load the N x T data
ndat = sio.loadmat('sample_dat.mat')['dat']
dfn = pd.DataFrame(ndat[0])
spk_arr = np.array(dfn['spikes'][:])

# Load Gpy for Gaussian Processes
packages = [package.project_name for package in pkg_resources.working_set]
if 'elephant' not in packages: pip.main(['install', 'elephant'])
from elephant.gpfa import GPFA
import quantities as pq
import neo as n

def gen_spk_train(spk_arr):
    """
    Generate a spike train for use in neo.SpikeTrain
    :param spk_arr: ndarray of dims trials x neurons x time
    :return spk_train: neo Spike Train object
    """
    spk_train = []
    for trial in spk_arr:
        spk_trial = []
        for neuron in trial:
            ntimes = np.where(neuron==1)[0] * pq.ms
            spk_trial.append(n.SpikeTrain(ntimes,t_stop=len(neuron)*pq.ms))
        spk_train.append(spk_trial)
    return spk_train

def plot_spk_train(spk_train,trial):
    '''
    Plot spike train for a specified trial.
    :param spk_train: neo SpikeTrain object
    :param trial: int, trial number to plot in [1, total number of trials]
    :return:
    '''
    for i, spiketrain in enumerate(spk_train[trial-1]):
        plt.plot(spiketrain, np.ones(len(spiketrain)) * i, ls='', marker='|')
    plt.title(f'Spike trains from trial {trial}')
    plt.tight_layout()
    plt.show()


def perform_gpfa(spk_train, bin_size=1 * pq.ms, latent_dimensionality=3):
    """
    Compute gpfa trajectories of a neo SpikeTrain object.
    :param spk_train: neo SpikeTrain object
    :param bin_size: quantities object, time bin width
    :param latent_dimensionality: dims of lower-dimensional trajectory
    :return trajs: ndarray, computed GPFA trajectories
    """
    gpfa_3dim = GPFA(bin_size=bin_size,x_dim=latent_dimensionality)
    trajs = gpfa_3dim.fit_transform(spk_train)
    return trajs


def plot_trajs(trajs):
    f = plt.figure(figsize=(6,5))
    ax = f.add_subplot(1, 1, 1, projection='3d'); c = 'rcbyg'
    [ax.plot(trajs[i][0],trajs[i][1],trajs[i][2],
             c[i%len(c)]) for i in np.arange(len(trajs))]
    avg_trajs=np.mean(trajs,axis=0); print(avg_trajs.shape)
    ax.plot(avg_trajs[0,:],avg_trajs[1,:],avg_trajs[2,:],c='k',
            linewidth=5.0, label='avg')
    ax.set_title('GPFA per-trial trajectories')
    ax.legend(loc='best', fontsize=10)
    ax.view_init(azim=-133, elev=41)
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.set_zlabel('z')
    plt.tight_layout()
    plt.show()