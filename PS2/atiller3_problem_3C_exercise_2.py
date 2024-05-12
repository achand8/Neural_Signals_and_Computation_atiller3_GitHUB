# Problem Set 2, #3C

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA as pca
from atiller3_problem_3B_exercise_2 import *


# Compute PCA components
def compute_pca_recon(x):
    x-= x.mean(axis=0,keepdims=True)
    pca_x = pca(n_components=x.shape[0]).fit(x)
    x_evs = pca_x.components_[:6,:]
    recon = np.dot(pca_x.transform(x)[:,:6],x_evs).T # dims neurons x time
    return recon


def plot_pca_err(X):
    err = np.array([np.linalg.norm(
        compute_pca_recon(X[i][0]) - X[i][0].T, axis=0)
        for i in np.arange(len(X))])
    plt.bar(np.arange(err.shape[1]), err.mean(axis=0));
    plt.xlabel('Time (bins)'); plt.ylabel('Error, norm')
    plt.title('Mean error of reconstruction over X, norm')
    plt.tight_layout()
    plt.show()
