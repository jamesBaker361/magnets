import gmsh
import numpy as np

def r_z(z):
    """Radius as a function of z."""
    return 0.1 + 0.02 * np.sin(2 * np.pi * z / 1.0)  # Example r(z)

def t_z(z):
    """Wire thickness as a function of z."""
    return 0.01 + 0.002 * np.cos(2 * np.pi * z / 1.0)  # Example t(z)

def n_z(z):
    """Turns per length as a function of z."""
    return 10 + 5 * np.sin(2 * np.pi * z / 1.0)  # Example n(z)

def generate_coil_mesh(L, num_slices=100):
    """
    Generate a coil mesh based on parameters t(z), r(z), and n(z).
    
    Parameters:
        L: Length of the solenoid
        num_slices: Number of slices along the z-axis
    """
    gmsh.initialize()
    gmsh.model.add("coil")

    dz = L / num_slices
    points = []
    wires = []

    for i in range(num_slices):
        z = i * dz
        r = r_z(z)
        t = t_z(z)
        n_turns = n_z(z)
        
        # Calculate helix points for this z slice
        theta = np.linspace(0, 2 * np.pi * n_turns * dz, num=20)
        for angle in theta:
            x = (r + t / 2) * np.cos(angle)
            y = (r + t / 2) * np.sin(angle)
            points.append(gmsh.model.geo.addPoint(x, y, z, meshSize=0.01))

        # Create lines to connect helix points
        num_points = len(points)
        for j in range(len(theta) - 1):
            wires.append(gmsh.model.geo.addLine(points[num_points - len(theta) + j],
                                                points[num_points - len(theta) + j + 1]))
    
    # Create physical group for visualization
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write("coil_mesh.msh")
    gmsh.finalize()

# Example usage
L = 1.0  # Length of the solenoid
generate_coil_mesh(L)
