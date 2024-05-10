# Problem Set 2, #1E

# Load tools
from atiller3_problem_1C_exercise_2 import *


# Vary A and plot tuning curve estimate and histogram of spikes
def plot_p1e(gHat, g, la, Avec, its, sigma_pvec,Mvec,N, method, draws=1):
    """
    Plotter for poiss_m(). Generates 8 (4x2) plots.
    """
    fig,ax = plt.subplots(4,2,figsize=[12,8])
    sup=4
    mse = [(np.square(gHat[i] - g[i]).sum() / len(g[i])) for i in range(len(g))]  # calculate mse
    for i in np.arange(sup):
        r_hist = np.array([np.random.poisson(m,draws) for m in la[i]]).T
        ax[i,1].hist(r_hist)
        ax[i,1].set(xlabel='Bins'); ax[i,1].set(ylabel='Counts')
        ax[i,1].set(title=f"Distribution: M = {Mvec[i]}, Strength={Avec[i]}, sig = {sigma_pvec[i]}")
        ax[i,0].plot(gHat[i], 'm', label='$\hat{g}$')
        ax[i,0].plot(g[i], 'k', label='g')
        ax[i,0].set_ylabel('Amplitude')
        ax[i,0].set_title("Estimated g: "+f"MSE = {mse[i]:.3f} for Iter: {its[i]}, M = {Mvec[i]},"+f" \nStrength={Avec[i]}, sig={sigma_pvec[i]}")
        ax[i,0].legend(loc='best', fontsize=8)
    fig.suptitle("$\hat{g}$ and distributions of r for "+f"N={(N)} \ngiven M,  A,"+" and tuning $\sigma$"+f" via method: {method}")
    fig.tight_layout(); plt.show()


def plot_1e_helper(Avec, sigma_pvec, prior='poisson', draws=100):
    N = 60; Mvec = [2080, 520, 260, 130]; its=0
    gHat,g,its,rl = map(list, zip(*[mle_g(N=N, M=Mvec[i], prior=prior,A=Avec[i],sigma_p=sigma_pvec[i]) for i in range(len(Avec))]))
    gHat = np.array(gHat,dtype='object'); g = np.array(g,dtype='object')
    X,grl = map(list,zip(*[rl_term for rl_term in rl]))
    la=np.array([np.exp(np.array((grl[i]).T.dot(np.array(X[i])))) for i in range(len(Avec))],dtype='object'); its=np.array(its)
    plot_p1e(gHat, g, la, Avec, its, sigma_pvec,Mvec,N, method=f'MLE, prior: {prior}',draws=draws)


# Plot MLE (gaussian)
# sigma_pvec = np.full((1,4),1e-8)[0]; Avec = [0.01, 0.1, 0.5, 1.0]
# plot_1e_helper(Avec,sigma_pvec)
#
# Avec = np.full((1,4),1.0)[0]; sigma_pvec = [2.0, 1e-2, 1e-4, 1e-8]
# plot_1e_helper(Avec,sigma_pvec)