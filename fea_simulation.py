from sfepy.discrete import Problem
from sfepy.mechanics.matcoefs import stiffness_from_lame
from sfepy.discrete.fem import Mesh,Field,FEDomain
from sfepy.base.base import Struct
from sfepy.discrete.common.domain import Domain
from sfepy.discrete.common.region import Region
from sfepy.base.conf import ProblemConf
from sfepy.discrete import (FieldVariable, Material, Problem, Function,
                                Equation, Equations, Integral)
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.base.base import IndexedStruct

import numpy as np
import sys

def fix_A_fun(ts, coors, bc=None, problem=None, extra_arg=None):
    return np.zeros_like(coors)

def solve_magnetic_field(mesh_file, point, current_density, permeability):
    """
    Solve the magnetostatic problem to find the magnetic field at a point.

    Parameters:
        mesh_file (str): Path to the SfePy mesh file.
        point (array-like): Coordinates of the point where the magnetic field is calculated.
        current_density (float): Magnitude of the current density (assumed uniform).
        permeability (float): Magnetic permeability of the material.
    """
    mesh=Mesh.from_file("tetra_mesh.vtk")
    domain=FEDomain("domain",mesh)
    dim = domain.shape.dim

    min_x, max_x = domain.get_mesh_bounding_box()[:,0]
    eps = 1e-8 * (max_x - min_x)

    omega = domain.create_region('Omega', 'all')
    gamma1 = domain.create_region('Gamma1',
                                  'vertices in x < %.10f' % (min_x + eps),
                                  'facet')
    gamma2 = domain.create_region('Gamma2',
                                  'vertices in x > %.10f' % (max_x - eps),
                                  'facet')

    field = Field.from_args('fu', np.float64, 'vector', omega,
                            approx_order=2)
    
    # Define field variables for magnetic vector potential (A)
    A = FieldVariable('A', 'unknown', field)  # Magnetic vector potential (unknown)
    v = FieldVariable('v', 'test', field, primary_var_name='A')  # Test function for A

    # Define the material properties (permeability)
    m = Material('m', mu=1.0)  # Permeability (assuming vacuum for simplicity)

    # Define the current density (for example, uniform current density)
    J = Material('J', val=[[0.02], [0.01]])  # Current density vector per unit volume

    bc_fun = Function('fix_A_fun', fix_A_fun, extra_args={'extra_arg': 'hello'})
    fix_A = EssentialBC('fix_A', gamma1, {'A.all': bc_fun})

    integral = Integral('i', order=3)
    # Define terms for the weak form
    # This term represents the weak form of the magnetostatics equation: ∇×(1/μ ∇×A) = J
    t1 = Term.new('dw_curl_lvf(m.mu, v, A)', integral, omega, m=m, v=v, A=A)

    # Current density source term (for example)
    t2 = Term.new('dw_volume_lvf(J.val, v)', integral, omega, J=J, v=v)

    # Equation for magnetostatics (balance equation)
    eq = Equation('balance', t1 + t2)
    eqs = Equations([eq])

    # Linear solver (direct solver)
    ls = ScipyDirect({})

    # Nonlinear solver setup (if needed, for nonlinearity)
    nls_status = IndexedStruct()
    nls = Newton({}, lin_solver=ls, status=nls_status)

    # Problem setup
    pb = Problem('magnetostatics', equations=eqs)
    pb.set_bcs(ebcs=Conditions([fix_A]))  # Apply boundary conditions
    pb.set_solver(nls)

    # Solve the problem (no need to save results if you're only interested in the magnetic field)
    state = pb.solve(save_results=False)

    # Save the solution to a VTK file (for visualization, optional)
    name ='test_magnetic_field.vtk'
    pb.save_state(name, state)

    

current_density=10
permeability=100

solve_magnetic_field("tetra_mesh.vtk",[2,2,2],current_density, permeability)