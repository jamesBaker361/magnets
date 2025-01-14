
'''

input parameters:
Length
n(z)=(n_0,n_1,,,) for sinusoidal/polynomial approximation 
r(z)=(r_0,r_1,,,)
wire diameter
current
x,y,z

output parameters:
material used
power dissipation
B strength at any point
'''


import time
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space in T·m/A

# Example solenoid parameters
def n(z):
    """Number of turns per unit length as a function of z (can be a constant or a function)."""
    # For example, a constant number of turns per unit length
    return 1000  # turns per meter

def r(z):
    """Radius of the solenoid as a function of z (can be a constant or a function)."""
    # For example, a constant radius
    return 0.01  # meters (1 cm)

def magnetic_field_contribution(z, x, y, n_func, r_func, I):
    """Calculate the magnetic field contribution at point (x, y, z) from a segment at z'."""
    r_z = r_func(z)
    n_z = n_func(z)
    
    # Calculate the distance from the segment at z' to the point (x, y, z)
    R = np.sqrt(x**2 + y**2 + (z - z)**2)  # Distance from the current element to the point
    
    # Biot-Savart law contribution (simplified for magnetic field along the z-axis)
    B_z = (mu_0 * n_z * I) / (2 * np.pi * R**2) * (x**2 + y**2)**0.5
    return B_z

def compute_magnetic_field(x, y, z, z1, z2, n_func, r_func, I):
    """Compute the total magnetic field at point (x, y, z) by integrating over the solenoid."""
    # Integrating along the length of the solenoid from z1 to z2
    B_z, _ = quad(magnetic_field_contribution, z1, z2, args=(x, y, n_func, r_func, I))
    return B_z

# Example usage
I = 1.0  # Current in amperes
z1, z2 = 0, 1  # Length of the solenoid (from z1 to z2)

# Point where the magnetic field is calculated
x, y, z = 0.005, 0.005, 0.5  # Point (x, y, z) in meters

# Compute the magnetic field at point (x, y, z)
start=time.time()
B = compute_magnetic_field(x, y, z, z1, z2, n, r, I)
end=time.time()
print(f"Magnetic field at point ({x}, {y}, {z}) is: {B} Tesla elapsed {end-start}")

