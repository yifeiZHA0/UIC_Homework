import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
a1 = 2
a2 = 1
Z = a1 * X + a2 * Y

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.Blues,
                       linewidth=0, antialiased=False)

ax.set_xlabel(r'$x_1$', fontsize=20, color='blue')
ax.set_ylabel(r'$x_2$', fontsize=20, color='blue')
ax.set_zlabel(r'$x_3$', fontsize=20, color='blue')
