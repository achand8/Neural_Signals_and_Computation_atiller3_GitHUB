"""
Exercise 3, Part 1
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

# Find n ROI
def ROI_mask(summ_img,n_ROI,tolerance=25,preview=False):
    """
    Output binary mask positions for n ROI from a 2D image.
    :param summ_img: ndarray, Image to apply ROI mask, dim: width_pixels x height_pixels
    :param tolerance: int, pixel tolerance for which to find individual ROI
    :param n_ROI: int, number of ROI to find
    :return: mask, list of arrays of coordinates of n ROI as (y_coordinate, x_coordinate) column vectors
    """
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
    col = ['r','c','m','y','b'] # intialize some colors
    if ( preview==True ):
        fig,ax = plt.subplots(1,2); axs = ax.ravel()
        [axs[i].imshow(summ_img,cmap='gray') for i in range(2)]
        axs[0].set_title('Original Summary Image'); axs[1].set_title(f"Mask Preview, number of ROI: {n_ROI}")
        [axs[1].plot(mask[i][:,0],mask[i][:,1],col[i]+'.') for i in np.arange(n_ROI)]
        plt.show()
    return mask

roi_mask = ROI_mask(summ_img,5,preview=True)

