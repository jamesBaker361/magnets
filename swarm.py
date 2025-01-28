from pyswarm import pso
import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import random
import string
from simsopt.field import ToroidalField, InterpolatedField
from simsopt.geo import CurveXYZFourier
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from simsopt.field import trace_particles_starting_on_curve,trace_particles
# NOTE: Most of the functions and classes implemented in the tracing
# NOTE: submodule can be imported directly from the field module
import numpy as np
from simsopt.geo import CurveXYZFourier
from simsopt.field import Current, Coil
from simsopt.field import BiotSavart
import matplotlib.pyplot as plt 
from simsopt.field.tracing import MinZStoppingCriterion, MaxRStoppingCriterion,MaxZStoppingCriterion
from torch.nn import Linear,Sequential
import gymnasium as gym
import numpy as np
from scipy.optimize import minimize
import torch
from static_globals import *

parser=argparse.ArgumentParser()
from simulation import evaluate_fourier

VELOCITY="velocity"
CONFINEMENT="confinement"

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="simulation")
parser.add_argument("--n_trials",type=int,default=1000)
parser.add_argument("--n_coils",type=int,default=4)
parser.add_argument("--max_fourier_mode",type=int,default=2)
parser.add_argument("--objective",type=str,default="velocity",help=f"{VELOCITY} or {CONFINEMENT}")
parser.add_argument("--nozzle_radius",type=float,default=0.1,help="nozzle radius for velocity")
parser.add_argument("--radius",type=float,default=1.0,help="chamber radius- maximum for particles, minimum for coils")
parser.add_argument("--n_particles",type=int,default=3)

if __name__=="__main__":
    args=parser.parse_args()

    start_height=float(Z_MAX-Z_MIN)*0.8
    start_positions=[[0,0,start_height],[0,0.1,start_height],[0,0.25,start_height]]

    stopping_criteria=[MaxRStoppingCriterion(args.radius),MinZStoppingCriterion(Z_MIN),MaxZStoppingCriterion(Z_MAX)]
    coeff_per_coil=2*(1+2*args.max_fourier_mode)+1
    total_coefficients=args.n_coils*coeff_per_coil #not including amps

    # Define an objective function
    def objective(coefficients):
        fourier_coefficients=coefficients[:-args.n_coils]
        amp_list=coefficients[-args.n_coils:]
        return evaluate_fourier(fourier_coefficients=fourier_coefficients,
                                max_fourier_mode=args.max_fourier_mode,
                                start_positions=start_positions,
                                start_velocities=[1 for _ in start_positions],
                                nozzle_radius=args.nozzle_radius,
                                objective=args.objective,
                                amp_list=amp_list,
                                stopping_criteria=stopping_criteria
                                )
    
    lower_bounds=[0 for _ in range(total_coefficients)]+[100 for _ in range(args.n_coils)]
    upper_bounds=[2.5 for _ in range(total_coefficients)]+[10000 for _ in range(args.n_coils) ]

    best=pso(objective,lb=lower_bounds, ub=upper_bounds,swarmsize=10)