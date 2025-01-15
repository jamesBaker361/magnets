import numpy as np
from scipy.integrate import tplquad,quad
import time
import numpy as np

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)
I = 1.0  # Current in the solenoid (A)

# Solenoid parameters
def r(z):
    return 0.05 + 0.01 * z  # Radius as a function of z (m)

def t(z):
    return 0.001 + 0.0002 * z  # Wire thickness as a function of z (m)

def n(z):
    return 1000 * np.exp(-0.1 * z)  # Turns per unit length as a function of z

# Biot-Savart integrand for each component of B
def biot_savart_integrand(r_wire, phi, z, rx, ry, rz, component):
    # Source position in cylindrical coordinates
    x_source = r_wire * np.cos(phi)
    y_source = r_wire * np.sin(phi)
    z_source = z

    # Relative vector
    r_vec = np.array([rx - x_source, ry - y_source, rz - z_source])
    r_mag = np.linalg.norm(r_vec)

    if r_mag == 0:
        return 0  # Avoid singularity at the observation point

    # Current density
    A_wire = np.pi / 4 * t(z)**2
    J_mag = I / A_wire
    J = J_mag * np.array([-np.sin(phi), np.cos(phi), 0])  # Azimuthal direction

    # Biot-Savart cross product
    cross = np.cross(J, r_vec)
    return cross[component] / r_mag**3

# Magnetic field computation at (rx, ry, rz)
def compute_magnetic_field(rx, ry, rz, L):
    B = np.zeros(3)  # Initialize magnetic field components

    for component in range(3):  # Integrate each component of B
        print(":)")
        result,_ = tplquad(
            lambda r_wire, phi, z: biot_savart_integrand(r_wire, phi, z, rx, ry, rz, component),
            0, L,  # z limits
            lambda z: 0, lambda z: 2 * np.pi,  # phi limits
            lambda z,p: r(z), lambda z,p: r(z) + t(z)  # r_wire limits
        )
        B[component] = mu_0 / (4 * np.pi) * result

    return B

def calculate_length(h,num_points=1000):
    def dr_dz(z):
        dz = h/ num_points
        return (r(z + dz) - r(z)) / dz

    def integrand(z):
        dr = dr_dz(z)
        r_val = r(z)
        n_val = n(z)
        return np.sqrt(1 + dr**2 + (r_val**2) * (2 * np.pi * n_val)**2)
    
    # Perform the integration
    length, _ = quad(integrand, 0, h)
    return length

# Example: Compute B at point (0.1, 0, 0.5) for a solenoid of length 1.0 m
h=1
L = calculate_length(h)  # Length of the solenoid (m)
rx, ry, rz = 0.1, 0.0, 0.5  # Field point (m)

def get_function(param_list,sinusoidal=True):

    def f(z):
        if sinusoidal:
            assert len(param_list)%2==0
            value=0
            for i in range(0,param_list,2):
                a=param_list[i]
                b=param_list[i+1]
                value+=a*np.cos(i *z)
                value+=b*np.sin(i*z)
        else:
            value=sum([param * z**degree for degree,param in enumerate(param_list)])
        return value

start=time.time()
B = compute_magnetic_field(rx, ry, rz, L)
end=time.time()
print(f"Magnetic field at ({rx}, {ry}, {rz}): Bx = {B[0]:.6e} T, By = {B[1]:.6e} T, Bz = {B[2]:.6e} T elapsed {end -start} seconds")
