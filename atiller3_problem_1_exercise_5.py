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

# Run PCA with, and view scree plot for features to search
# Reshape transformed data into images x time
n_comp = 8
PCA_model = PCA(n_components=n_comp, whiten=True, random_state = 0)
PCA_trfm = PCA_model.fit_transform(M_PCA)
PCA_comp_imgs = PCA_trfm.reshape(M.shape[0],M.shape[0],-1)

# View scree plot
pca_var = PCA_model.explained_variance_
fig1,ax1 = plt.subplots(1,2, figsize=(9,6)); axs1 = ax1.ravel()
axs1[0].plot(pca_var); axs1[0].set_title('Scree plot, first eigenvector\n '
                                       'dominates')
axs1[1].plot(pca_var[1:10]); axs1[1].set_title('Scree plot without first eigenvector, \n'
                                             '3 components seem adequate')
plt.show()

#  Select a transform with ROI mask candidates by manual search over first 4 PCs (Components 2 works here)
fig2,ax2 = plt.subplots(2,4,figsize=(11,6)); axs2=ax2.ravel()
for i in range(n_comp):
    axs2[i].imshow(PCA_comp_imgs[:,:,i],cmap='gray')
    axs2[i].set_title(f"Component {i+1}")
plt.show()

# Component 3 has at least two interesting ROI in one component, so let's plot them
# Use the method from 4.1
summ_img = PCA_comp_imgs[:,:,2] # Take a PCA result as a summary image
tolerance = 20; n_ROI = 2;
summ_img2 = summ_img.copy()
summ_img2 += abs(summ_img2.min()) # scale the summary image
summ_img2 = (np.round(summ_img2/summ_img2.max())).astype('int') # binarize the summary image
loc_roi=np.flip(np.array(np.where(summ_img2==1)),axis=0); N = loc_roi.shape[1]
loc_roi_cand, loc_roi_n, mask = [],[],[]

# Group "1s" into ROI
for j in np.arange(N-1):
    xydist = abs(loc_roi[:,j]-loc_roi[:,j+1])
    if (xydist.max()>tolerance):
        loc_roi_n.append(np.array(loc_roi_cand)); loc_roi_cand = []
    else:
        loc_roi_cand.append(loc_roi[:,j])
roi_len = np.argsort([-len(k) for k in loc_roi_n])
[mask.append(loc_roi_n[roi_len[i]]) for i in range(n_ROI)]

# For looking at ROI, optional
col = ['r', 'c', 'm', 'y', 'b']  # intialize some colors
fig3, ax3 = plt.subplots(1, 2);
axs3 = ax3.ravel()
[axs3[i].imshow(summ_img, cmap='gray') for i in range(2)]
axs3[0].set_title('Original Summary Image');
axs3[1].set_title(f"Mask Preview, number of ROI: {n_ROI}")
[axs3[1].plot(mask[i][:, 0], mask[i][:, 1], col[i] + '.') for i in np.arange(n_ROI)]
plt.show()

# Compute time traces
# Using the masks as indexers on the original video M, form a time trace per ROI
# Try a rudimentary normalization using quantile. Trends and other fluorescence factors such as bleaching are ignored
roi_time = np.array([np.quantile(M[:,mask[i][:,0],mask[i][:,1]],q=.8,axis=1) for i in range(n_ROI)])
fig4,ax4 = plt.subplots(n_ROI,1, figsize=(11,9)); axs4 = ax4.ravel()
for i in range(n_ROI):
    axs4[i].plot(roi_time[i],c=col[i])
    axs4[i].set_title(f"ROI {i+1}")
fig4.suptitle(rf"{n_ROI} ROI over time")
plt.show()
