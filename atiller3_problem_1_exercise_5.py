"""
Exercise 5, Part 1
"""

# import tools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the movie
# Import tools
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as io

# Load movie as array
mov = 'TEST_MOVIE_00001-small.tif'  # movie should be in the same directory
mov_frames = io.imread(mov)  # shape: t x width_pixels x height_pixels
M = np.array(mov_frames).astype('float64')  # M for movie

# Reshape data to (n_samples x n_features) for sklearn
M_PCA = M.reshape((M.shape[0],-1)).T

# Run PCA, and view scree plot for features to keep
PCA_model = PCA(whiten=True)
M_PCA_run = PCA_model.fit_transform(M_PCA)

# View scree plot
pca_var = PCA_model.explained_variance_
fig,ax = plt.subplots(1,4); axs = ax.ravel()
axs[0].plot(pca_var); axs[0].set_title('Scree plot, first eigenvector\n '
                                       'dominates and is likely trivial')
axs[1].plot(pca_var[1:10]); axs[1].set_title('Scree plot without first eigenvector, \n'
                                             '3 components seem adequate.')
# Reconstruct the ROIs using PCA. Use the second component onward and add to subplots
def PCA_img_recon(PCA_proj, PCA_modl, n_keep, dropFirst=True):
    """
    Recostruct image from PCA using n_keep number of components.
    :param PCA_proj: ndarray, PCA projection
    :param PCA_modl: model obj, fitted PCA model
    :param n_keep: keeps up to n_keep number of components. All others will be set to zero.
    :param dropFirst: drop the first component, if trivial
    :return: PCA_recon, ndarray of PCA reconstruction using inverse transform method
    """
    dims = PCA_proj.shape[1]
    PCA_recon = PCA_proj.copy(); PCA_recon[:,n_keep+1:] = 0;
    if (dropFirst==True): # Drop the first component, if trivial
        PCA_recon[:,0] = 0
    PCA_recon = PCA_modl.inverse_transform(PCA_recon)
    PCA_recon = PCA_recon.reshape(dims,dims,-1)
    return PCA_recon


recon1 = PCA_img_recon(M_PCA_run, PCA_model, 1)
recon3 = PCA_img_recon(M_PCA_run,PCA_model, 3)
axs[2].imshow(recon1.sum(axis=2), cmap='gray_r'); axs[2].set_title('Second component'
                                                                        ' only')
axs[3].imshow(recon3.sum(axis=2), cmap='gray_r'); axs[3].set_title('Second - fourth components'
                                                                     ' kept')
plt.show()

print('If I choose more PCs, the contrast between the ROIs and background increases.')