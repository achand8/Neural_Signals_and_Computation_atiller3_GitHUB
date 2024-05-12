# Problem Set 2, #3E

# Load tools
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import jPCA
from jPCA.util import load_churchland_data, plot_projections


# Load the N x T data
edat = sio.loadmat('exampleData.mat')['Data'][0]
reach27 = edat[26][0]; x=reach27

# Run jPCA
import urllib.request
data_url = 'https://github.com/nwb4edu/development/blob/0f181c8092d79278fcb0320d9f53bc33fbb0df85/exampleData.mat?raw=true'
# Get the data and save it locally as "sleep_data.txt"
path, headers = urllib.request.urlretrieve(data_url, './exampleData.mat')
edat, times = load_churchland_data(path)

jpca = jPCA.JPCA(num_jpcs=6)
# times = np.linspace(-50,550,61)
projected, full_data_var, pca_var_capt, jpca_var_capt = jpca.fit(edat, times=times, tstart=-50, tend=550)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_projections(projected, axis=axes[0], x_idx=0, y_idx=1) # Plot the first jPCA plane
plot_projections(projected, axis=axes[1], x_idx=2, y_idx=3) # Plot the second jPCA plane
plt.show()
