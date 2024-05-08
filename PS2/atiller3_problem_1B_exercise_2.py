# Problem Set 2, #1B

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.optimize import leastsq
from atiller3_problem_1A_exercise_2 import poiss_m


# Least-squares solution of r = Xg + e
# for g takes the form of   X† r = g
def lsq_g(N,M,alpha=5):
    """
    Generate least squares solution of r = Xg + e
    :return: gHat, an estimate of g using least-squares solution
    :return: g, actual g of linearized system r = Xg + e
    """
    _,X,g = poiss_m(N=N,M=M,alpha=alpha,draws=1)
    r = X.T @ g + np.random.normal(0,1,M)
    Xpinv = np.linalg.pinv(X @ X.T) @ X
    gHat = (Xpinv @ r).T
    return gHat,g


def plot_p1b(gHat, g, Nvec, Mvec, method):
    mse = [(np.square(gHat[i] - g[i]).sum() / len(g[i])) for i in range(len(g))] # calculate mse
    fig,ax = plt.subplots(4,3,figsize=[12,10]); axs = ax.ravel()
    sup=12;
    for i in np.arange(sup):
        axs[i].plot(gHat[i],'m',label='$\hat{g}$')
        axs[i].plot(g[i],'k',label='g')
        axs[i].set_xlabel('N'); axs[i].set_ylabel('Amplitude')
        axs[i].set_title(f"MSE = {mse[i]:.3f} for N={Nvec[i]}, M={Mvec[i]}")
        axs[i].legend(loc='best', fontsize=8)
    fig.suptitle(r'$\hat{g}$ vs g for M=N, M=2N, M=$\frac{N}{2}$ pairs via '+method)
    fig.tight_layout(); plt.show()


# Nvec = [10, 10, 10, 50, 50, 50, 100, 100, 100, 200, 200, 200]
# Mvec = [10, 20, 5, 50, 100, 25, 100, 200, 50, 200, 400, 100]
# gHat,g = map(list, zip(*[lsq_g(N=Nvec[i], M=Mvec[i]) for i in range(len(Nvec))]))
# gHat = np.array(gHat,dtype='object'); g = np.array(g,dtype='object')
# plot_p1b(gHat,g, Nvec, Mvec, method='least squares')