import numpy as np
from sfepy.discrete.fem import Mesh

import numpy as np
from sfepy.discrete.fem.mesh import Mesh

def generate_coil_mesh_tetrahedral(n_func, t_func, r_func, z_min, z_max, num_points, radial_divisions=10, height_divisions=10):
    """
    Generate a 3D tetrahedral coil mesh given n(z), t(z), and r(z).

    Parameters:
        n_func (function): Turns per unit length as a function of z.
        t_func (function): Thickness as a function of z.
        r_func (function): Radius as a function of z.
        z_min (float): Minimum z-value.
        z_max (float): Maximum z-value.
        num_points (int): Number of points to discretize along the z-axis.
        radial_divisions (int): Number of radial divisions for coil cross-section.
        height_divisions (int): Number of height divisions for coil volume.

    Returns:
        Mesh: SfePy Mesh object representing the coil.
    """
    # Discretize the z-axis
    z_values = np.linspace(z_min, z_max, num_points)

    # Initialize node list
    nodes = []
    elements = []

    # Generate coil geometry
    for i, z in enumerate(z_values[:-1]):
        dz = z_values[i + 1] - z
        n_turns = n_func(z) * dz
        radius = r_func(z)
        thickness = t_func(z)

        # Number of points per turn
        points_per_turn = 100
        theta_values = np.linspace(0, 2 * np.pi * n_turns, int(points_per_turn * n_turns))

        # Generate nodes for the cross-section
        for theta in theta_values:
            for r_frac in np.linspace(0, 1, radial_divisions):
                r_current = radius + r_frac * thickness
                x = r_current * np.cos(theta)
                y = r_current * np.sin(theta)
                nodes.append([x, y, z])

    # Convert nodes to array
    nodes = np.array(nodes)
    num_nodes = len(nodes)

    # Create tetrahedral elements
    for i in range(num_nodes - height_divisions - 1):
        # Connect nodes in a tetrahedral pattern
        elements.append([i, i + 1, i + height_divisions, i + height_divisions + 1])

    # Convert elements to array
    elements = np.array(elements)
    mat_ids = [np.ones(len(elements), dtype=int)]

    # Create SfePy mesh
    mesh = Mesh.from_data(
        "coil_mesh",
        nodes,  # Nodes need to be transposed to (dim, n_nod)
        None,
        [elements],
        mat_ids,
        ["3_4"],  # Tetrahedral elements
    )
    return mesh


def generate_coil_mesh(n_func, t_func, r_func, z_min, z_max, num_points):
    """
    Generate a 3D coil mesh given n(z), t(z), and r(z).
    
    Parameters:
        n_func (function): Turns per unit length as a function of z.
        t_func (function): Thickness as a function of z.
        r_func (function): Radius as a function of z.
        z_min (float): Minimum z-value.
        z_max (float): Maximum z-value.
        num_points (int): Number of points to discretize along the z-axis.
    
    Returns:
        Mesh: SfePy Mesh object representing the coil.
    """
    # Discretize the z-axis
    z_values = np.linspace(z_min, z_max, num_points)
    
    # Initialize lists for coil nodes and elements
    nodes = []
    elements = []
    
    # Generate coil geometry
    for i, z in enumerate(z_values[:-1]):
        dz = z_values[i + 1] - z
        n_turns = n_func(z) * dz
        radius = r_func(z)
        thickness = t_func(z)
        
        # Number of points per turn
        points_per_turn = 100
        theta_values = np.linspace(0, 2 * np.pi * n_turns, int(points_per_turn * n_turns))
        
        # Generate points for the current segment
        segment_nodes = []
        for theta in theta_values:
            x = (radius + thickness / 2) * np.cos(theta)
            y = (radius + thickness / 2) * np.sin(theta)
            segment_nodes.append([x, y, z])
        
        # Append nodes to the global list
        nodes.extend(segment_nodes)
        
        # Create elements (optional: use line segments or surface elements)
        if i > 0:
            num_prev_nodes = len(nodes) - len(segment_nodes)
            for j in range(len(segment_nodes) - 1):
                elements.append([num_prev_nodes + j, num_prev_nodes + j + 1])
    
    # Convert lists to arrays
    nodes = np.array(nodes)
    elements = np.array(elements)
    print(nodes.shape)
    print(elements.shape)
    mat_ids = [np.ones(len(elements), dtype=int)]

    
    # Create SfePy mesh
    mesh = Mesh.from_data(
        "coil_mesh",
        nodes,
        None,
        [elements],
       mat_ids,
        ["1_2"],  # Line elements (1D); for 3D, use "3_4" for tetrahedra.
    )
    return mesh

# Example functions
n = lambda z: 10 + z  # Linear increase in turns per unit length
t = lambda z: 0.1     # Constant thickness
r = lambda z: 1 + 0.5 * np.sin(z)  # Varying radius with sine wave

# Generate the coil mesh
coil_mesh = generate_coil_mesh_tetrahedral(n, t, r, z_min=0, z_max=1, num_points=10)

# Save the mesh for SfePy
coil_mesh.write("coil_mesh.vtk")
