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
NMF_model = NMF(n_components=n_comp)
NMF_trfm = NMF_model.fit_transform(M_NMF) # W, transformed data, W matrix
NMF_comp_imgs = NMF_trfm.reshape(M.shape[0],M.shape[0],-1)

#  Select a transform with ROI candidates by manual search (Component  works here)
for i in range(n_comp):
    plt.imshow(NMF_comp_imgs[:,:,i],cmap='gray')
    plt.show()

#
# # Reconstr
# dims = M.shape[0]
# NMF_recon = M_NMF_proj.copy()
# # NMF_recon[:, n_keep + 1:] = 0
# # NMF_recon = NMF_model.inverse_transform(NMF_recon)
# NMF_recon = NMF_recon.reshape(dims, dims, -1)
# NMF_recon += 1e-6
# # NMF_recon /= NMF_recon.max(axis=2)
#
# #
# # NMF
# plt.title('(Somewhat) highlighted ROIs via NMF, components=20')
# plt.imshow(NMF_recon[:,:,:-4].sum(axis=2),cmap='gray')
# plt.show()

print('There is not a dependence on rank as there was for PCA -- as the number of components\n'
      'increases for NMF (until roughly 40 components here), rank can decrease, which is very \n'
      'different from the linearly independent basis/columns assumption of components as in PCA.')