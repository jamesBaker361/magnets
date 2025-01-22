import numpy as np
from sfepy.discrete.fem import Mesh

import numpy as np

def discretize_prism_to_tetrahedra(base_vertices, top_vertices):


    # Define tetrahedra
    tetrahedra = [
        [0, 1, 2, 5],  # A, B, C, C'
        [0, 5, 3, 1],  # A, C', A', B
        [1, 5, 4, 3]   # B, C', B', A'
    ]

    nodes=base_vertices+top_vertices
    elements=[]
    for indices in tetrahedra:
        elements.append([nodes[i] for i in indices])

    return elements

nodes=[[0,0,0],[0,1,0], [0.5,1,0], [0,0,1],[0,1,1], [0.5,1,1]]
elements=discretize_prism_to_tetrahedra([0,1,2],[3,4,5])

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

mesh.write("tetra_mesh.vtk")