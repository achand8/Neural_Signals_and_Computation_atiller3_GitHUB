# Problem Set 2, #2A

# Load tools
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import pkg_resources
import pip

# Load Gpy for Gaussian Processes
packages = [package.project_name for package in pkg_resources.working_set]
if 'elephant' not in packages: pip.main(['install', 'elephant'])
from elephant.gpfa import GPFA
import quantities as pq
import neo as n

# Load the N x T data
ndat = sio.loadmat('sample_dat.mat')['dat']

dfn = pd.DataFrame(ndat[0])
spk_arr = np.array(dfn['spikes'][:])
psth = spk_arr.sum(axis=0)
plt.bar(np.arange(len(psth.T)),np.sum(psth,axis=0).T)
plt.show()

def gprocess(spk_dat):
    for trial in spk_dat:
        for neuron in trial:
        mu = neuron.mean(axis=0)
        var = (spk_dat-mu).T @ (spk_dat-mu) / spk_dat.shape[0]
    return var

spk_train = []
for trial in spk_arr:
    spk_trial = []
    for neuron in trial:
        ntimes = np.where(neuron==1)[0] * pq.ms
        spk_trial.append(n.SpikeTrain(ntimes,t_stop=len(neuron)*pq.ms))
    spk_train.append(spk_trial)

for i, spiketrain in enumerate(spk_train[0]):
    plt.plot(spiketrain, np.ones(len(spiketrain)) * i, ls='', marker='|')

plt.tight_layout()
plt.show()

bin_size = 1 * pq.ms
latent_dimensionality = 3
gpfa_3dim = GPFA(bin_size=bin_size,x_dim=latent_dimensionality)
trajs = gpfa_3dim.fit_transform(spk_train)

f = plt.figure(figsize=(15,5))
ax = f.add_subplot(1, 1, 1, projection='3d')
# avg_trajs=np.mean(trajs,axis=0)
[ax.plot(trajs[i][0],trajs[i][1],trajs[i][2]) for i in np.arange(len(trajs))]
ax.view_init(azim=-100, elev=0)
ax.set_xlim3d(-0.05,0.05)
ax.set_ylim3d(-0.07,0.06)
ax.set_zlim3d(-0.02,0.8)
plt.tight_layout()
plt.show()









