# Problem Set 2, #1B

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from atiller3_problem_1A_exercise_2 import *

# Least-squares solution of r = Xg + e
# for g takes the form of   X† r = g
def lsq_g(N,M):
    noise = np.full((1,M),1) @ np.eye(M) # initialize variance of noise as 1
    r,X,g = poiss_m(M=M,N=N,trunc2=True)
    Xpinv = np.linalg.pinv(X @ X.T) @ X
    gHat = -Xpinv @ r
    return gHat,g


def plot_p1b(gHat,g):
    plt.figure(figsize=(10,6))
    plt.plot(gHat,'m',label='estimated g')
    plt.plot(g,'k',label='g')
    plt.legend(loc='best')
    plt.show()

gHat,g = lsq_g(200,250)
plot_p1b(gHat,g)