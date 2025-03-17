import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a mesh grid of x2 and x3 values
x2 = np.linspace(-10, 10, 100)
x3 = np.linspace(-10, 10, 100)
x2, x3 = np.meshgrid(x2, x3)  # Create a 2D grid

# Create the first plane: x1 + x2 + x3 = 0, i.e., x1 = -x2 - x3
x1_plane1 = -x2 - x3

# Create the second plane: x1 + 2x2 + 2x3 = 0, i.e., x1 = -2x2 - 2x3
x1_plane2 = -2*x2 - 2*x3

# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the first plane
ax.plot_surface(x2, x3, x1_plane1, alpha=0.5, rstride=100, cstride=100, color='r', label='Plane 1: x1 + x2 + x3 = 0')

# Plot the second plane
ax.plot_surface(x2, x3, x1_plane2, alpha=0.5, rstride=100, cstride=100, color='b', label='Plane 2: x1 + 2x2 + 2x3 = 0')

# Plot the line of intersection (x1=0, x2=x3)
# The line will be where x1 = 0, and x2 = x3
x3_line = np.linspace(-10, 10, 100)
ax.plot(x3_line, x3_line, np.zeros_like(x3_line), color='g', label='Intersection line: x1 = 0, x2 = x3')

# Labels and title
ax.set_xlabel('x2')
ax.set_ylabel('x3')
ax.set_zlabel('x1')
ax.set_title('Intersection of Two Planes')

# Show the plot
ax.legend()
plt.show()
