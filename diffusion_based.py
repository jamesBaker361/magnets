#use diffusion to generate parameters- reward function is based on whether electrons do what they're supposed to
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.optimize import minimize_scalar
from stable_baselines3 import PPO
import time
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

AMPS=1000
m = PROTON_MASS
q = ELEMENTARY_CHARGE
Ekin = 10*ONE_EV

def calculate_reward(observation:list,nozzle_radius:int):
    reward=0.0
    counts=0
    for [x,y,z,v_x,v_y,v_z] in observation:
        if np.linalg.norm([x,y])< nozzle_radius and z>=1:
            reward+=v_z #for each particle, that is in the nozzle, we want as much z momentumas possible
            counts+=1
    #print("found rewards")
    return reward,counts

def evaluate_fourier(fourier_coefficients:list,
                     max_fourier_n:int,
                     start_positions:list,
                     start_velocities:list,
                     stopping_criteria:list,
                     nozzle_radius:float):
    n_coils=len(fourier_coefficients)
    coil_list=[]
    for f_c in fourier_coefficients:
        curve = CurveXYZFourier(1000, max_fourier_n)
        all_fourier=np.concatenate(f_c, [0 for 0 in range(2*max_fourier_n)-1])
        curve.x=all_fourier
        coil = Coil(curve, Current(AMPS)) 
        coil_list.append(coil)

    field=BiotSavart(coil_list)
    res_tys,res_phi_hits=trace_particles(field,start_positions,start_velocities,mass=m,charge=q,Ekin=Ekin,mode="full",forget_exact_path=True,
                                                stopping_criteria=stopping_criteria)
    
    observation=[rt[-1][1:] for rt in res_tys]
    reward,counts=calculate_reward(observation,nozzle_radius)

    return reward

