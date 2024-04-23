# """
# Exercise 1, Part 2
# """
#

# From the video in Part 1, we see that frames
# (3,9), (6,29), (8,27) seem particularly offset.

# # Import tools
import numpy as np
from numpy.fft import ifftshift, ifft2, fft2, fftshift
import imageio.v2 as io
from scipy import signal
import matplotlib.pyplot as plt
from atiller3_problem_1_exercise_1 import load_tif

# Load the .tif
mov = "TEST_MOVIE_00001-small-motion.tif" # movie should be in same directory
mov_frames = io.imread(mov) # shape: t x width_pixels x height_pixels
M = np.array(mov_frames).astype('float64')  # M for movie

# Create a function for producing correlations, and run a few
def corr2d(frame1,frame2,M,mean_flag=False):
    """
    Calculate the correlation between two frames.
    :param frame1: int, frame 1
    :param frame2: int, frame 2
    :param M: ndarray, video of dimensions time x width_pixels x height_pixels
    :param mean_flag: anny
    :return: correlation between the two frames
    """
    img=M[frame1,:,:].astype('float64')
    temp=M[frame2,:,:].astype('float64'); zpad = len(img)+len(temp)-1
    if mean_flag==True: # Apply rudimentary preprocessing
        img-=img.mean(); img=img/(len(img)*img.std())
        temp-=temp.mean(); temp=temp/(len(temp)*temp.std())
    temp = temp[::-1,::-1] # Perform the correlation
    corr1 = ifft2(fft2(img, (zpad,zpad)) * fft2(temp, (zpad,zpad)))
    return corr1.real

# Plot the correlations
fig,ax = plt.subplots(2,3,figsize=(9, 6))
corr_n = [corr2d(3,9,M), corr2d(6,29,M), corr2d(8,27,M),
            corr2d(3,9,M,mean_flag=True), corr2d(6,29,M,mean_flag=True),
                corr2d(8,27,M,mean_flag=True)]
framep = np.array([[3,9],[6,29],[8,27]])
axs = ax.ravel()
for i in np.arange(6):
    axs[i].set_title(f"Frame pair {framep[np.mod(i,3),:]}")
    axs[i].imshow(corr_n[i],cmap='gray')
fig.suptitle('Correlations, raw (top row) and mean-centered (bottom row)')
plt.show()