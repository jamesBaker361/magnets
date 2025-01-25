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

parser=argparse.ArgumentParser()

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


# Objective function to minimize the change in coefficients
def objective(params,original_coeffs):
    return np.sum((params - original_coeffs)**2)

# Parameterized Fourier functions
def fourier(theta, coeffs,mode):
    total=coeffs[0]
    for m in range(1,mode+1):
        total+=coeffs[(2*m)-1]*np.cos(m*theta)+coeffs[(2*m)]*np.sin(m*theta)
    return total


def r(theta, x_coeffs, y_coeffs,mode):
    x_val = fourier(theta, x_coeffs,mode)
    y_val = fourier(theta, y_coeffs,mode)
    return np.sqrt(x_val**2 + y_val**2)

def random_vector_in_cylinder(radius):
    # Generate a random angle between 0 and 2π
    theta = np.random.uniform(0, 2 * np.pi)
    
    # Generate a random radius using sqrt for uniform distribution in 2D
    r = np.sqrt(np.random.uniform(0, 1)) * radius
    
    # Generate a random height between 0 and 1
    z = np.random.uniform(0, 1)
    
    # Convert polar to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.array([x, y, z])

def get_fixed_fourier(initial_guess,r_min,theta_vals,mode):
    # Constraint: r(theta) >= r_min for all theta
    def radius_constraint(params):
        midpoint = len(params) // 2
        x_coeffs=params[:midpoint]
        y_coeffs=params[midpoint:]
        return [r(theta, x_coeffs, y_coeffs,mode) - r_min for theta in theta_vals]
    constraints = [{'type': 'ineq', 'fun': radius_constraint}]

    # Optimize
    result = minimize(objective, x0=initial_guess, args=(initial_guess) ,constraints=constraints)
    print(result)
    print(dir(result))

    return result.x

#uses simsopt to simulate particles to build dataset

def evaluate_fourier(fourier_coefficients:list,
                     max_fourier_mode,
                     start_positions,
                     start_velocities,
                     stopping_criteria,
                     nozzle_radius,
                     objective:str,
m = PROTON_MASS,
q = ELEMENTARY_CHARGE,
Ekin = 10*ONE_EV,
AMPS=1000):
        n_coils=len(fourier_coefficients)
        coil_list=[]
        for f_c in fourier_coefficients:
            curve = CurveXYZFourier(1000, max_fourier_mode)
            #print(f_c)
            #print([0 for _ in range(2*max_fourier_mode -1)])
            all_fourier=np.concatenate((f_c, [0 for _ in range((2*max_fourier_mode))]))
            curve.x=all_fourier
            coil = Coil(curve, Current(AMPS)) 
            coil_list.append(coil)

        field=BiotSavart(coil_list)
        res_tys,res_phi_hits=trace_particles(field,np.array(start_positions),np.array(start_velocities),mass=m,charge=q,Ekin=Ekin,mode="full",forget_exact_path=True,
                                                    stopping_criteria=stopping_criteria)
        
        observations=[rt[-1] for rt in res_tys]
        rewards=[]
        for [t,x,y,z,v_x,v_y,v_z] in observations:
            if objective==VELOCITY:
                if np.linalg.norm([x,y])<= nozzle_radius and z>=1:
                    rewards.append(v_z) #for each particle, that is in the nozzle, we want as much z momentumas possible
                    counts+=1
            #print("found rewards")
            elif objective==CONFINEMENT:
                rewards.append(t)

        return rewards,observations


def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    n_trials=args.n_trials
    data_folder=f"{args.objective}_simulation_data"
    os.makedirs(data_folder,exist_ok=True)
    random_letters = ''.join(random.choices(string.ascii_letters, k=5))

    stopping_criteria=[MaxRStoppingCriterion(args.radius),MinZStoppingCriterion(0),MaxZStoppingCriterion(1)]
    coeff_per_coil=2*(1+2*args.max_fourier_mode)+1
    total_coefficients=args.n_coils*coeff_per_coil

    beginning=time.time()
    theta_vals = np.linspace(0, 2 * np.pi, 100)
    with open(os.path.join(data_folder, f"{random_letters}.csv"),"w+") as file:
        file.write(f"{total_coefficients},{args.n_coils},{args.radius},{args.max_fourier_mode},{args.objective},{args.nozzle_radius},{args.n_particles}\n")
        for trial in range(n_trials):
            
            initial_coefficients= np.random.uniform(0,1,(args.n_coils, coeff_per_coil))
            coefficients=[]
            for initial_guess in initial_coefficients:
                result=get_fixed_fourier(initial_guess[:-1],args.radius,theta_vals,args.max_fourier_mode)
                #print("initial guess -1",initial_guess[-1])
                coefficients.append(np.concatenate((result,[initial_guess[-1]] )))
            start_positions=[]
            start_velocities=[]
            for v in range(args.n_particles):
                start_positions.append(random_vector_in_cylinder(args.radius))
                start_velocities.append(np.random.uniform(0,1))
            
            rewards,observations=evaluate_fourier(coefficients,args.max_fourier_mode,start_positions,
                                                start_velocities,
                                                stopping_criteria,
                                                args.nozzle_radius,
                                                args.objective)
            
            for r,o,vector,velocity in zip(rewards,observations,start_positions,start_velocities):
                print("o",o)
                print("r",r)
                print("vector",vector)
                print("velocity",velocity)
                file.write(",".join(map(str, r+o+vector+velocity))+"\n")
            

            
            
            
            


    

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")