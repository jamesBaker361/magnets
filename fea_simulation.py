from sfepy.discrete import Problem
from sfepy.mechanics.matcoefs import stiffness_from_lame
from sfepy.discrete.fem import Mesh,Field,FEDomain
from sfepy.base.base import Struct
from sfepy.discrete.common.domain import Domain
from sfepy.discrete.common.region import Region
from sfepy.base.conf import ProblemConf
import numpy as np
import sys

def solve_magnetic_field(mesh_file, point, current_density, permeability):
    """
    Solve the magnetostatic problem to find the magnetic field at a point.

    Parameters:
        mesh_file (str): Path to the SfePy mesh file.
        point (array-like): Coordinates of the point where the magnetic field is calculated.
        current_density (float): Magnitude of the current density (assumed uniform).
        permeability (float): Magnetic permeability of the material.
    """
    # Load mesh
    mesh = Mesh.from_file(mesh_file)
    domain=FEDomain("domain",mesh)
    

    # Define regions
    regions = {
        'Omega': 'all',
        'Gamma': ('vertices of surface', 'facet'),
    }

    omega_region=domain.create_region("Omega","all")

    # Define field for vector potential A
    field = Field.from_args(
        'vector_potential', np.float64, 'vector', omega_region,approx_order=1
    )

    fields={'vector_potential': ('real', 'vector', 'Omega', 2),}

    # Define material properties
    materials = {
        'm': {
            'permeability': permeability,
        },
        'j': {
            'current_density': np.array([0, 0, current_density]),
        },
    }

    # Define equations
    equations = {
        'balance_of_forces': """
            dw_lin_elastic.m.Omega(A, v) = dw_volume_lvf.j.Omega(J, v)
        """,
    }

    conf=ProblemConf.from_dict({
        'regions': regions,
        'fields': fields,
        'materials': materials,
        'equations': equations,
        'variables': {
            'A': ('unknown field', 'vector_potential', 0),
            'v': ('test field', 'vector_potential', '0'),
        }},sys.modules[__name__])

    # Define problem
    problem = Problem.from_conf(conf)

    # Solve problem
    problem.solve()

    # Extract solution (magnetic vector potential)
    vec_potential = problem.get_variables()['A'].get_state()

    # Compute magnetic field B = curl(A)
    bfield = problem.evaluate('ev_curl.A', vec_potential)

    # Find the magnetic field at the desired point
    from scipy.spatial import KDTree
    kdtree = KDTree(mesh.coors)
    closest_idx = kdtree.query(point)[1]
    magnetic_field_at_point = bfield[closest_idx]

    return magnetic_field_at_point

current_density=10
permeability=100

solve_magnetic_field("tetra_mesh.vtk",[2,2,2],current_density, permeability)