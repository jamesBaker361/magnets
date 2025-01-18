import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define r(z), t(z), and n(z)
def r(z):
    return 1 + 0.1 * np.sin(z)  # Example: radius varies with sine wave

def t(z):
    return 0.05 + 0.01 * np.cos(z)  # Example: thickness varies with cosine wave

def n(z):
    return 5 + z * 0.5  # Example: turns per length increases linearly

# Generate coil geometry
def generate_coil(z_min, z_max, num_pointsr,r=r,n=n):
    z = np.linspace(z_min, z_max, num_points)  # Height of the coil
    theta = np.cumsum(2 * np.pi * n(z) * np.gradient(z))  # Angle for turns
    x = r(z) * np.cos(theta)  # X-coordinates
    y = r(z) * np.sin(theta)  # Y-coordinates
    return x, y, z

if __name__=="__main__":
    # Generate the coil path
    z_min, z_max, num_points = 0, 10, 1000
    x, y, z = generate_coil(z_min, z_max, num_points)

    # Plot the coil
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=2, label="Coil Path")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Coil Model")
    plt.legend()
    output_file = "3d_coil_model.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")  # Save with high resolution
    print(f"Figure saved as {output_file}")