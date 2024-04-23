"""
Exercise 4, Part 1
"""

# Import tools
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as io

# Load movie as array
mov = 'TEST_MOVIE_00001-small.tif'  # movie should be in the same directory
mov_frames = io.imread(mov)  # shape: t x width_pixels x height_pixels
M = np.array(mov_frames).astype('float64')  # M for movie

# Use an approach from 2.2 to create a summary image
M2=M.copy()
M2[M2<=.5*M2.max()] = -1e-6  # make small values constant to prevent biasing by Gaussian noise
M2 /= M2.max(axis=0); M2 += abs(M2.min(axis=0))+1e-6  # scale, make nonzero so probability behaves
M_prob = M2/M2.sum(axis=0, keepdims=True)  # compute probability
M_entr = -(-M_prob*np.log2(M_prob)).sum(axis=0)  # compute entropy
summ_img=M_entr

# Find n ROI from 3.1
tolerance = 25; n_ROI = 5;
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

# Compute time traces
# Using the masks as indexers on the original video M, form a time trace per ROI
roi_time = np.array([np.quantile(M[:,mask[i][:,0],mask[i][:,1]],q=.8,axis=1) for i in range(n_ROI)])

# Try a rudimentary df/F, where F0 is the median. Other fluorescence effects such as fluorescence
# trend changes are ignored
col = ['r', 'c', 'm', 'y', 'b']  # intialize some colors
fig,ax = plt.subplots(5,1, figsize=(11,9)); axs = ax.ravel()
for i in range(n_ROI):
    axs[i].plot(roi_time[i],c=col[i])
    axs[i].set_title(f"ROI {i+1}")
fig.suptitle(rf"{n_ROI} ROI over time, color-coded to ROI of 3.1")
plt.show()