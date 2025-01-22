import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the phi list
def get_phi_list(z_max, steps, n):
    output = [0]
    z_max = float(z_max)
    z_step = z_max / steps
    for z in range(1,steps):
        norm_z = z * z_step
        output.append((2 * np.pi * n(norm_z) / steps)+output[-1])
    return output


z_max = 2
# Define the n(z) function (example: linear function)
def n(z):
    if z<z_max//2:
        return 2
    return 10  # Example: turns increase linearly with z

# Parameters
R = 1  # Constant radius

steps =1000

# Generate phi values
phi_list = get_phi_list(z_max, steps, n)

# Generate z values
z_values = np.linspace(0, z_max, steps)

# Convert cylindrical to Cartesian coordinates
x_values = R * np.cos(phi_list)
y_values = R * np.sin(phi_list)

# Plot the points
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_values, y_values, z_values, c=z_values, cmap='viridis', marker='o')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points in Cylindrical Coordinates')

# Save the figure
fig.savefig("cylindrical_points_plot.png", dpi=300)
plt.close(fig)  # Close the figure to free resources
