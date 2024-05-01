# Problem Set 2, #1A


# Load tools
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian


# Generate samples from a Poisson distribution
def poiss_m(N, alpha=5, M=1, trunc=False, trunc2=False):
    """
    :param M: int, dimension of stimuli
    :param N: int, points to
    :param alpha: float, width factor for Gaussian distribution
    :param k: int, support of poisson [0,k-1]
    :return r_pmf:
    """
    sup = N; std = (N - 1) / (2 * alpha)
    g = gaussian(N,std) * np.cos(2*np.pi*(np.arange(N))/10).T
    X = 2 * np.random.rand(N,M)
    la = np.exp(g.dot(X)).reshape(M,1)
    if trunc:
        sup = 6; la = la[:sup]
    k = np.arange(sup)
    if trunc2: k=[0]
    k_fact = np.array([math.factorial(num) for num in k]).ravel()
    r_pmf = np.power(la,k) * np.exp(-la) / k_fact
    return r_pmf, X, g


def plot_p1a(N, M, alpha=5, trunc=False, trunc2=False):
    if trunc: sup = 6;
    if trunc2: exit()
    else: sup=N
    fig,ax = plt.subplots(2,3,figsize=[10,5]); axs = ax.ravel()
    for i in np.arange(sup):
        r,_,_ = poiss_m(N=N, M=M,alpha=alpha, trunc=trunc, trunc2=trunc2)
        axs[i].hist(r[i,:],bins=np.arange(0,sup,1)-.5,density=True);
        axs[i].set_xlabel('Counts'); axs[i].set_ylabel('p')
    fig.suptitle(f"Poisson distributions of random draws of x for N={N}, k up to {sup-1}")
    fig.tight_layout()
    plt.show()

# poiss_m(N=58,M=10,trunc2=True)