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
from torch.nn import Linear,Sequential

AMPS=1000
m = PROTON_MASS
q = ELEMENTARY_CHARGE
Ekin = 10*ONE_EV

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("max_fourier_mode",type=int,default=1)
parser.add_argument("n_coils",type=int,default=4)
parser.add_argument("epochs",type=int,default=10)
parser.add_argument("n_layers",type=int,default=4)
parser.add_argument("max_fourier_mode",type=int,default=2)
parser.add_argument("residuals",action="store_true")
parser.add_argument("increasing",action="store_true")


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
                     max_fourier_mode:int,
                     start_positions:list,
                     start_velocities:list,
                     stopping_criteria:list,
                     nozzle_radius:float):
    n_coils=len(fourier_coefficients)
    coil_list=[]
    for f_c in fourier_coefficients:
        curve = CurveXYZFourier(1000, max_fourier_mode)
        all_fourier=np.concatenate(f_c, [0 for 0 in range(2*max_fourier_mode)-1])
        curve.x=all_fourier
        coil = Coil(curve, Current(AMPS)) 
        coil_list.append(coil)

    field=BiotSavart(coil_list)
    res_tys,res_phi_hits=trace_particles(field,start_positions,start_velocities,mass=m,charge=q,Ekin=Ekin,mode="full",forget_exact_path=True,
                                                stopping_criteria=stopping_criteria)
    
    observation=[rt[-1][1:] for rt in res_tys]
    reward,counts=calculate_reward(observation,nozzle_radius)

    return reward

class Denoiser(torch.nn.Module):
    def __init__(self, n_features:int, n_layers:int,residuals:bool,increasing:bool):
        super().__init__()
        self.n_features=n_features
        self.n_layers=n_layers
        self.residuals =residuals

        diff=n_features/n_layers
        down_layer_list=[]
        up_layer_list=[]
        prev=n_features
        for n in range(n_layers//2):
            if increasing:
                current=prev+diff
            else:
                current=prev-diff
            down_layer_list.append(Linear(prev,current))
            prev=current
        
        if increasing:
            self.mid_block=Linear(prev,n_features*2)
            prev=n_features*2
        else:
            self.mid_block=Linear(prev,n_features//2)
            prev=n_features//2
        

        for n in range(n_layers//2):
            if increasing:
                current=prev-diff
            else:
                current=prev+diff
            up_layer_list.append(Linear(prev,current))
            prev=current

        self.down_block=Sequential(*down_layer_list)
        self.up_block=Sequential(*up_layer_list)

    def forward(self,x):
        if self.residuals:
            residual_list=[]
            for linear in self.down_block:
                x=linear(x)
                residual_list.append(x)
            x=self.mid_block(x)
            residual_list=residual_list[::-1]
            print("len res list",len(residual_list))
            for i,linear in enumerate(self.up_block):
                x=linear(x)+residual_list[i]
        else:
            x=self.down_block(x)
            x=self.mid_block(x)
            x=self.up_block(x)
        return x




def main(args):
    n_features=args.n_coils+args.n_coils*(1+2*args.max_fourier_mode)
    denoiser=Denoiser(n_features,args.n_layers,args.residuals,args.increasing)
    model_parameters=[p for p in denoiser.parameters()]
    print(f"optimzing {len(model_parameters)} model params")
    optimizer=torch.optim.adamw.AdamW(model_parameters)


    return

if __name__=="__main__":
    args=parser.parse_args()
    main(args)