import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))

import implicators as imp

# Create a grid of values
a_vals = np.linspace(0, 1, 200)
b_vals = np.linspace(0, 1, 200)
A, B = np.meshgrid(a_vals, b_vals)

# Vectorized computation of the implicator
Z = np.vectorize(imp.imp_reichenbach)(A, B)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with colormap
surf = ax.plot_surface(A, B, Z, cmap='hot', edgecolor='none')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Labels and title
ax.set_title("Gödel Implicator $I(a, b)$")
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_zlabel("I(a, b)")

plt.tight_layout()
plt.show()
