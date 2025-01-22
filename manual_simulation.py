import numpy as np

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)

# Wire parameters
L = 1.0                  # Length of the wire (m)
I = 1.0                  # Current through the wire (A)

# Spiral parameters
def r_z(z):
    return 0.05 + 0.01 * z  # Radius varies linearly with z

def n_z(z):
    return 10 + 2 * z       # Turns per unit length increases with z

# Observation point
r_obs = np.array([0.1, 0.1, 0.1])  # Point where B is calculated (x, y, z)

# Discretization parameters
num_z_segments = 200        # Number of segments along the wire
num_radial_segments = 10    # Number of segments along the radial direction
num_angular_segments = 20   # Number of segments along the angular direction

# Discretize the wire along z
z_vals = np.linspace(-L/2, L/2, num_z_segments)

# Initialize magnetic field
B = np.zeros(3)  # Bx, By, Bz

# Perform integration
for zi in z_vals:
    # Compute the position of the spiral at z
    r_i = r_z(zi)
    n_i = n_z(zi)
    x_prime = r_i * np.cos(2 * np.pi * n_i * zi)
    y_prime = r_i * np.sin(2 * np.pi * n_i * zi)
    z_prime = zi
    r_prime = np.array([x_prime, y_prime, z_prime])
    
    # Compute derivatives for dL
    drdz_x = -2 * np.pi * n_i * r_i * np.sin(2 * np.pi * n_i * zi) + (0.01 * np.cos(2 * np.pi * n_i * zi))  # Chain rule
    drdz_y = 2 * np.pi * n_i * r_i * np.cos(2 * np.pi * n_i * zi)+  (10 parammetic dm scales ref)..
