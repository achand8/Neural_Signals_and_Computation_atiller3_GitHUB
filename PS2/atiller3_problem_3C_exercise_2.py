# Problem Set 2, #3C

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA as pca
from atiller3_problem_3B_exercise_2 import *

edat = sio.loadmat('exampleData.mat')['Data'][0]
reach27 = edat[26][0].T; x=reach27

# Compute PCA components
x-= x.mean(axis=1,keepdims=True)
pca_x = pca(n_components=x.shape[0]).fit(x)
x_evs = pca_x.components_[:6,:]

recon = np.dot(pca_x.transform(x)[:,:6],x_evs).T

# compute error here

print(recon.shape)