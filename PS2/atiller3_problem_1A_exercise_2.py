# Problem Set 2, #1A

# Load tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian


# Generate samples from a Poisson distribution
def poiss_m(N, alpha=5, M=1, draws=1, A=1):
    """
    :param N: int, # of samples
    :param M: int, # of stimuli
    :param alpha: float, width factor for Gaussian distribution
    :param draws: int, # of response draws from Poisson distribution
    :return r_pmf:
    """
    std = (N - 1) / (2 * alpha)
    g = gaussian(N,std) * np.cos(2*np.pi*(np.arange(N))/10).T
    X = 2 * np.random.rand(N,M) * A
    la = np.exp(g.dot(X))
    r_pmf = np.array([np.random.poisson(m,draws) for m in la]).T
    return r_pmf, X, g


def plot_p1a(N, M, alpha=5, draws=1):
    """
    Plotter for poiss_m(). Generates 6 (2x3) plots.
    """
    fig,ax = plt.subplots(2,3,figsize=[10,5]); axs = ax.ravel()
    sup=6
    for i in np.arange(sup):
        r,_,_ = poiss_m(N=N, M=M, alpha=alpha, draws=draws)
        axs[i].hist(r)
        axs[i].set_xlabel('Bin'); axs[i].set_ylabel('Counts')
    fig.suptitle(f"Distribution of r for $x_{M}$ with N={N}, draws={draws}")
    fig.tight_layout(); plt.show()

# plot_p1a(N=100,M=2,draws=100)