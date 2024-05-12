# Problem Set 2, #2D

# Load tools
import numpy as np
import matplotlib as mpl
from atiller3_problem_2C_exercise_2 import *


# Create a gradient to test a simple hypothesis that trials
# adjacent to each other have similar trajectories perhaps
# gradient credit: Markus Dutschke
def cgrad(color1,color2,grad=0.5):
    color1=np.array(mpl.colors.to_rgb(color1))
    color2=np.array(mpl.colors.to_rgb(color2))
    return mpl.colors.to_hex((1-grad)*color1 + grad*color2)


def plot_trajs_grad(trajs,color1,color2):
    f = plt.figure(figsize=(10,12))
    ax = f.add_subplot(projection='3d')
    [ax.plot(trajs[i][0],trajs[i][1],trajs[i][2],
             c=cgrad(color1,color2,grad=i/trajs.shape[0]
                   )) for i in np.arange(len(trajs))]
    ax.set_title('GPFA per-trial trajectories - Gradient View - X vs Y')
    ax.view_init(azim=-180, elev=-90)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.show()

# Generate spike train (involves GPFA package, elephant)
# spk_train = gen_spk_train(spk_arr)
#
# # Plot a trial for sanity check
# plot_spk_train(spk_train, trial=8)
#
# # Compute GPFA and plot trajectories
# trajs = perform_gpfa(spk_train, bin_size=1 * pq.ms, latent_dimensionality=3)
# color1 = 'b'; color2 = 'r'
# plot_trajs_grad(trajs,color1,color2)