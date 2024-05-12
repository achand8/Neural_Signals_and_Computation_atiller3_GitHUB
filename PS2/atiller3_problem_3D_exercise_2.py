# Problem Set 2, #3D

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA as pca
from atiller3_problem_3C_exercise_2 import *


# Compute PCA components
def compute_pca_evs(x):
    x-= x.mean(axis=0,keepdims=True)
    pca_x = pca(n_components=x.shape[0]).fit(x)
    x_evs = pca_x.components_[:6,:]
    return x_evs


def plot_evs(x):
    """
    Plot the PCs associated with each dataset of the jPCA datasets
    :param x_evs: ndarray, PCs from PCA performed on one of the jPCA conditions
    :return:
    """
    x_evs = compute_pca_evs(x)
    plt.plot(x_evs[0,:],x_evs[1,:])

