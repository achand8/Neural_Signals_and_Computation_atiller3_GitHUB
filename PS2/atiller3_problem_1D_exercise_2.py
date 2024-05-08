# Problem Set 2, #1D

# Load tools
import numpy as np
import matplotlib.pyplot as plt
from atiller3_problem_1C_exercise_2 import *


# Plot MLE (gaussian or poisson prior)
for prior in ['gauss','poisson']:
    Nvec = [10, 10, 10, 50, 50, 50, 100, 100, 100, 200, 200, 200]
    Mvec = [10, 20, 5,  50, 100,25, 100, 200, 50,  200, 400, 100]
    prior = prior
    gHat,g,its,_ = map(list, zip(*[mle_g(N=Nvec[i], M=Mvec[i], prior=prior) for i in range(len(Nvec))]))
    gHat = np.array(gHat,dtype='object'); g = np.array(g,dtype='object'); np.array(its)
    plot_p1c(gHat,g, its, Nvec, Mvec, method=f'MLE, prior: {prior}')