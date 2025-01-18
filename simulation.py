import numpy as np
from scipy.integrate import tplquad,quad
import time
import numpy as np
import os
import random
import string
import argparse

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)
I = 1.0  # Current in the solenoid (A)


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
def compute_magnetic_field(rx, ry, rz, L,r,n,t):
    B = np.zeros(3)  # Initialize magnetic field components

    for component in range(3):  # Integrate each component of B

        result,_ = tplquad(
            lambda r_wire, phi, z: biot_savart_integrand(r_wire, phi, z, rx, ry, rz, component),
            0, L,  # z limits
            lambda z: 0, lambda z: 2 * np.pi,  # phi limits
            lambda z,p: r(z), lambda z,p: r(z) + t(z)  # r_wire limits
        )
        B[component] = mu_0 / (4 * np.pi) * result

    return B

def calculate_length(h,n,num_points=1000):
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



def get_function(param_list,sinusoidal=False):

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

    return f

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--n_trials",type=int,default=1000)
    parser.add_argument("--degree",type=int,default=6)
    parser.add_argument("--sinusoidal",action="store_true")

    args=parser.parse_args()

    n_trials=args.n_trials
    data_folder="simulation_data"
    os.makedirs(data_folder,exist_ok=True)
    random_letters = ''.join(random.choices(string.ascii_letters, k=5))

    degree=args.degree
    sinusoidal=args.sinusoidal
    beginning=time.time()
    with open(os.path.join(data_folder, f"{random_letters}.csv"),"w+") as file:
        file.write(f"{degree},{sinusoidal}\n")

        for trial in range(n_trials):
            #get random points
            x,y,z=[random.uniform(0,1) for _ in range(3)]

            r_list=[random.uniform(0,0.5) for _ in range(degree)]
            n_list=[random.uniform(0,100) for _ in range(degree)]
            t_list=[random.uniform(0,0.01) for _ in range(degree)]

            r=get_function(n_list)
            n=get_function(r_list)
            t=get_function(t_list)

            h=1
            L = calculate_length(h,n)  # Length of the solenoid (m)

            start=time.time()
            B = compute_magnetic_field(x,y,z, L,r,n,t)
            end=time.time()
            elapsed=round(end-start,3)
            line=",".join(map(str,[elapsed,L,x,y,z,B[0],B[1],B[2]]+r_list+n_list+t_list))
            file.write(f"{line}\n")
            #print(f"Magnetic field at ({rx}, {ry}, {rz}): Bx = {B[0]:.6e} T, By = {B[1]:.6e} T, Bz = {B[2]:.6e} T elapsed {end -start} seconds")
    print(f"all done time elapsed {end-beginning}")