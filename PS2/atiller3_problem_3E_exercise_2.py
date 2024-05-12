# Problem Set 2, #3E

# Load tools
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import jPCA
from jPCA.util import load_churchland_data, plot_projections
import urllib.request

# Import data from the source
data_url = 'https://github.com/nwb4edu/development/blob/0f181c8092d79278fcb0320d9f53bc33fbb0df85/exampleData.mat?raw=true'
path, headers = urllib.request.urlretrieve(data_url, './exampleData.mat')
edat, times = load_churchland_data(path)

# Intialize jPCA model
jpca = jPCA.JPCA(num_jpcs=6)

# This next approach is directly from the toolbox to obtain projections along jPC dimensions
# source: https://github.com/bantin/jPCA/blob/master/README.md
projected, full_data_var, pca_var_capt, jpca_var_capt = jpca.fit(edat, times=times, tstart=-50, tend=200)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_projections(projected, axis=axes[0], x_idx=0, y_idx=1) # Plot the first jPCA plane
axes[0].set_title('First jPC plane'); axes[0].set_xlabel('jPC1'); axes[0].set_ylabel('jPC2')
plot_projections(projected, axis=axes[1], x_idx=2, y_idx=3) # Plot the second jPCA plane
axes[1].set_title('Second jPC plane'); axes[1].set_xlabel('jPC2'); axes[1].set_ylabel('jPC3')
plt.show()

# Plot eigenvalue contributions
xpts = np.arange(6)+1
plt.plot(xpts,pca_var_capt,label='PCA'); plt.plot(xpts,jpca_var_capt,'m',label='jPCA')
plt.ylabel('Eigenvalue contribution'); plt.title('Eigenspectra contribution, PCA vs jPCA')
plt.xlabel('Eigenvalue'); plt.legend(loc='best')
plt.show()