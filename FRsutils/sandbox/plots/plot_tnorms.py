import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import FRsutils.core.tnorms as tn

# Create a grid of values
n = 200  # Replace with your desired length (must be even)
arr = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

similarity_vals = np.linspace(0, 1, n)
b_vals = arr
AVals, simVals = np.meshgrid(similarity_vals, b_vals)

tnrm = tn.TNorm.create("product")
tnrm = tn.TNorm.create("yager", p=0.15)

print(tnrm.name)

TNVals = tnrm.__call__(AVals, simVals)


# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with colormap
# surf = ax.plot_surface(A, B, Z, cmap='hot', edgecolor='none')
# Plot the points
ax.scatter(AVals, simVals, TNVals, c='blue', marker='.')

# Add color bar
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Labels and title
# Generate string like "yager_p_3.0"
title = tnrm.name
params = tnrm.describe_params_detailed()
param_str = " ".join(f" {k}={v['value']}" for k, v in params.items())
title = f"{title} ({param_str})" if param_str else title

ax.set_title(title)
ax.set_ylabel("similarity (x,y)")
ax.set_xlabel("A(y)")
ax.set_zlabel("T(similarity (x,y), A(y))")


plt.tight_layout()
plt.show()
