import numpy as np
import ffc
from dolfinx import mesh
from simulation import calculate_length, get_function
domain = mesh.create_unit_square( 8, 8, mesh.CellType.quadrilateral)