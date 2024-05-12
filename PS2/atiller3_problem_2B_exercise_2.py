# Problem Set 2, #2B

# Load tools
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as pca
from atiller3_problem_2A_exercise_2 import *


# Load the N x T data
ndat = sio.loadmat('sample_dat.mat')['dat']
dfn = pd.DataFrame(ndat[0])
spk_arr = np.array(dfn['spikes'][:])
psth = spk_arr.sum(axis=0) # Generate PSTH

# Generate GP-smoothed PSTH
gp_smooth = np.array([gp_smoothing(A=1.0,l=2e-3,x=psth[i,:]) for i in np.arange(psth.shape[0])])

# Perform PCA on PSTH
psth_X = psth - psth.mean(axis=1,keepdims=True)
pca_psth = pca(n_components=psth.shape[0]).fit(psth_X)
psth_evs = pca_psth.components_[:3,:]

# Perform PCA on smoothed PSTH
gpsth_X = gp_smooth - gp_smooth.mean(axis=1,keepdims=True)
pca_gpsth = pca(n_components=gp_smooth.shape[0]).fit(gpsth_X)
gpsth_evs = pca_gpsth.components_[:3,:]

def pc3d(x,x_sm,titles):
    '''
    Plot 2 3-dim ndarrays representing PCs side-by-side.
    :param x: ndarray of dims 3xN, left plot
    :param x_sm: ndarray of dims 3xN, right plot
    :param titles: list of 2 strings, become titles for subplots
    :return:
    '''
    f = plt.figure(figsize=(12, 5))
    ax = f.add_subplot(1, 2, 1, projection='3d');
    ax.plot(x[0,:], x[1,:], x[2,:], 'c'); ax.set_title(f'{titles[0]}')
    ax.view_init(azim=-35, elev=30); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax2 = f.add_subplot(1, 2, 2, projection='3d'); ax2.view_init(azim=-35, elev=30)
    ax2.plot(x_sm[0,:], x_sm[1,:], x_sm[2,:], 'c'); ax2.set_title(f'{titles[1]}')
    ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2'); ax2.set_zlabel('PC3')
    plt.show()
