import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='bwr', s=100)
ax.set_xlim(0, 4)
ax.set_ylim(0, 3.5)
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()