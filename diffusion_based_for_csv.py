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
import gymnasium as gym
import numpy as np
import csv
from modeling import set_to_one_hot,float_handle_na
from diffusers import DDIMScheduler
from diffusers.models.embeddings import TimestepEmbedding,LabelEmbedding
from unet_1D_with_class import Unet1DModelCLassy
import random

AMPS=1000
m = PROTON_MASS
q = ELEMENTARY_CHARGE
Ekin = 10*ONE_EV

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--csv_file",type=str,default="AFMPDT_database.csv")
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--n_layers",type=int,default=4)
parser.add_argument("--residuals",action="store_true")
parser.add_argument("--increasing",action="store_true")
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--batches_per_epoch",type=int,default=20)
parser.add_argument("--denoise_steps",type=int,default=10)
parser.add_argument("--gradient_accumulation_steps",type=int,default=2)
parser.add_argument("--num_timesteps",type=int,default=1000)


def calculate_reward(observation:list,nozzle_radius:int):
    reward=0.0
    counts=0
    for [x,y,z,v_x,v_y,v_z] in observation:
        if np.linalg.norm([x,y])< nozzle_radius and z>=1:
            reward+=v_z #for each particle, that is in the nozzle, we want as much z momentumas possible
            counts+=1
    #print("found rewards")
    return reward,counts


        

class Denoiser(torch.nn.Module):
    def __init__(self, n_features:int, n_layers:int,residuals:bool,increasing:bool,num_classes:int):
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
            
            down_layer_list.append(Linear(int(prev),int(current)))
            prev=current
        
        if increasing:
            self.mid_block=Linear(int(prev),int(n_features*2))
            prev=n_features*2
        else:
            self.mid_block=Linear(int(prev),int(n_features//2))
            prev=n_features//2
        

        for n in range(n_layers//2):
            if increasing:
                current=prev-diff
            else:
                current=prev+diff
            up_layer_list.append(Linear(int(prev),int(current)))
            prev=current

        self.down_block=Sequential(*down_layer_list)
        self.up_block=Sequential(*up_layer_list)
        self.time_embedding_model=TimestepEmbedding(n_features,1)

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
                
                x=linear(x +residual_list[i])
        else:
            x=self.down_block(x)
            x=self.mid_block(x)
            x=self.up_block(x)
        return x




def main(args):
    
    input_data=[]
    thrust_data=[]
    thruster_class_data=[]
    propellant_class_set=set()
    A_mat_class_set=set()
    C_mat_class_set=set()
    config_class_set=set()
    thruster_class_set=set()
    with open(args.csv_file,"r",encoding="cp1252") as file:
        dict_reader=csv.DictReader(file)
        for d_row in dict_reader:
            propellant_class_set.add(d_row["propellant"])
            A_mat_class_set.add(d_row["A_mat"])
            C_mat_class_set.add(d_row["C_mat"])
            config_class_set.add(d_row["config"])
            thruster_class_set.add(d_row["thruster"])
        propellant_dict=set_to_one_hot(propellant_class_set)
        A_mat_dict=set_to_one_hot(A_mat_class_set)
        C_mat_dict=set_to_one_hot(C_mat_class_set)
        config_dict=set_to_one_hot(config_class_set)
        thruster_dict=set_to_one_hot(thruster_class_set)
    with open(args.csv_file,"r",encoding="cp1252") as file:
        reader = csv.reader(file)
        first_row = next(reader)
        data=[]
        for row in reader:
            data.append(row)
        random.shuffle(data)
        for row in data:
            quantitative=[float_handle_na(d) for d in row[:14]]
            T_tot,J,B_A,mdot,error,Ra,Rc,Ra0,La,Rbi,Rbo,Lc_a,V,Pb=quantitative
            qualitative=row[14:-1]
            propellant,source,thruster,A_mat,C_mat,config=qualitative
            new_row=[J,B_A,Ra,Rc,Ra0,La,Rbi,Rbo,Lc_a,V]
            new_row=np.concatenate((new_row, propellant_dict[propellant], A_mat_dict[A_mat], C_mat_dict[C_mat], config_dict[config]))
            input_data.append(new_row)
            thruster_class_data.append(thruster_dict[thruster])
        

    n_features=len(new_row)
    print(f"data has {n_features} features")
    denoiser=Unet1DModelCLassy(in_channels=n_features,out_channels=n_features,
        use_timestep_embedding=True,use_class_embedding=True,num_classes=len(thruster_class_set))
    model_parameters=[p for p in denoiser.parameters()]
    print(f"optimzing {len(model_parameters)} model params")
    optimizer=torch.optim.AdamW(model_parameters)
    denoiser(torch.randn((1,n_features)))
    scheduler=DDIMScheduler(num_train_timesteps=args.timesteps)

    for e in range(args.epochs):
        for b in range(args.batches_per_epoch):
            input_batch=input_data[b]
            class_labels=thruster_class_data[b]
            optimizer.zero_grad()
            noise=torch.randn((args.batch_size,n_features))

            timesteps = torch.randint(0, args.timesteps, (args.batch_size,)).long()
            
            


        

if __name__=="__main__":
    args=parser.parse_args()
    main(args)