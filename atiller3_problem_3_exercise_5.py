"""
Exercise 5, Part 2
"""

# import tools
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as io
from sklearn.decomposition import FastICA

# Load movie as array
mov = 'TEST_MOVIE_00001-small.tif'  # movie should be in the same directory
mov_frames = io.imread(mov)  # shape: t x width_pixels x height_pixels
M = np.array(mov_frames).astype('float64')  # M for movie

# Reshape data to (n_samples x n_features) for sklearn
M_ICA = M.reshape((M.shape[0],-1)).T

# Run ICA, and reshape transforms to images x time
n_comp = 5
ICA_model = FastICA(n_components=n_comp, random_state=0, whiten='unit-variance')
ICA_trfm = ICA_model.fit_transform(M_ICA)
ICA_comp_imgs = ICA_trfm.reshape(M.shape[0],M.shape[0],-1)

#  Select a transform with ROI mask candidates by manual search (Component 14 works here)
fig2,ax2 = plt.subplots(1,5,figsize=(7,6)); axs2=ax2.ravel()
for i in range(n_comp):
    axs2[i].imshow(ICA_comp_imgs[:,:,i],cmap='gray_r')
    axs2[i].set_title(f"Component {i+1}")
plt.show()