"""
Exercise 5, Part 2
"""

# import tools
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as io
from sklearn.decomposition import NMF

# Load movie as array
mov = 'TEST_MOVIE_00001-small.tif'  # movie should be in the same directory
mov_frames = io.imread(mov)  # shape: t x width_pixels x height_pixels
M = np.array(mov_frames).astype('float64')  # M for movie

# Reshape data to (n_samples x n_features) for sklearn
M_NMF = M.reshape((M.shape[0],-1)).T

# Run NMF on the data and reshape transforms to images x time
n_comp = 12
NMF_model = NMF(n_components=n_comp, random_state=0)
NMF_trfm = NMF_model.fit_transform(M_NMF) # W, transformed data, W matrix
NMF_comp_imgs = NMF_trfm.reshape(M.shape[0],M.shape[0],-1)

#  Select a transform with ROI mask candidates by manual search (Component 1 works here)
fig2,ax2 = plt.subplots(3,4,figsize=(7,6)); axs2=ax2.ravel()
for i in range(n_comp):
    axs2[i].imshow(NMF_comp_imgs[:,:,i],cmap='gray')
    axs2[i].set_title(f"Component {i+1}")
plt.show()