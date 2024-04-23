# """
# Exercise 2, Part 1
# """
#

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as io

# Load movie as array
mov = 'TEST_MOVIE_00001-small.tif'  # movie should be in the same directory
mov_frames = io.imread(mov)  # shape: t x width_pixels x height_pixels
M = np.array(mov_frames)  # M for movie

# Plot the summary images as heatmaps
fig , ax = plt.subplots(1,3); axs = ax.ravel()
M_stats = [M.mean(axis=0), np.median(M,axis=0), M.var(axis=0)]
M_label = ['Mean', 'Median', 'Variance']
axs = ax.ravel()
for i in np.arange(3):
    axs[i].set_title(f"Summary Image: {M_label[i]}")
    axs[i].imshow(M_stats[i],cmap='gray')
fig.suptitle('Summary image, per pixel')
plt.show()