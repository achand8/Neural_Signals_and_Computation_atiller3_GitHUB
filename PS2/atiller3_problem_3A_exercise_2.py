# Problem Set 2, #3A

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


edat = sio.loadmat('exampleData.mat')['Data'][0]
reach27 = edat[26][0]
f = plt.figure(figsize=(4,8))
plt.imshow(reach27.T,cmap='jet')
plt.title('Heatmap of Condition 27')
plt.xlabel('Time (bin)'); plt.ylabel('Neuron')
plt.tight_layout()
plt.show()