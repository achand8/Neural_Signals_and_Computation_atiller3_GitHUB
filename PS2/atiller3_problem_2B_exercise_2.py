# Problem Set 2, #2A=B

# Load tools
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from atiller3_problem_2A_exercise_2 import *


# Load the N x T data
ndat = sio.loadmat('sample_dat.mat')['dat']
dfn = pd.DataFrame(ndat[0])
spk_arr = np.array(dfn['spikes'][:])
psth = spk_arr.sum(axis=0)

gp_smooth = gp_smoothing(A=1.0,2e-3,psth)
