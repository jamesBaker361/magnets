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
from modeling import batchify
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

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
parser.add_argument("--timesteps",type=int,default=1000)
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="magnet_diffusion")
parser.add_argument("--minimize",action="store_true")
parser.add_argument("--inference_steps",type=int,default=10)
parser.add_argument("--optimization_samples",type=int,default=10)
#parser.add_argument("--pretraining_penalty",action="store_true",help="whether to do the penalty during pretraining")




    



def main(args):

    



    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb"
    )
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    
    input_data=[]
    thrust_data=[]
    config_class_data=[]
    propellant_class_set=set()
    A_mat_class_set=set()
    C_mat_class_set=set()
    config_class_set=set()
    config_class_set=set()
    with open(args.csv_file,"r",encoding="cp1252") as file:
        dict_reader=csv.DictReader(file)
        for d_row in dict_reader:
            propellant_class_set.add(d_row["propellant"])
            A_mat_class_set.add(d_row["A_mat"])
            C_mat_class_set.add(d_row["C_mat"])
            config_class_set.add(d_row["config"])
            config_class_set.add(d_row["thruster"])
        propellant_dict=set_to_one_hot(propellant_class_set)
        A_mat_dict=set_to_one_hot(A_mat_class_set)
        C_mat_dict=set_to_one_hot(C_mat_class_set)
        config_dict=set_to_one_hot(config_class_set)
        config_dict=set_to_one_hot(config_class_set)
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
            n_quantitative_inputs=len(new_row)
            new_row=np.concatenate((new_row, propellant_dict[propellant], A_mat_dict[A_mat], C_mat_dict[C_mat]))
            input_data.append(new_row)
            config_class_data.append(config_dict[config])
        

    class SquaredMagnitudePenalty(torch.nn.Module):
        def __init__(self):
            super(SquaredMagnitudePenalty, self).__init__()

        def forward(self, input):
            # Extract the first 10 values from each row
            first_n = input[:, :n_quantitative_inputs]
            
            # Compute the squared magnitude (sum of squares for each row)
            squared_magnitude = torch.sum(first_n ** 2, dim=1)
            
            # Return the mean penalty across the batch
            return torch.mean(squared_magnitude)
        

    sm_penalty=SquaredMagnitudePenalty()
    def calculate_penalty(batch:torch.Tensor,minimize):
        if minimize:
            penalty=sm_penalty(batch)
        return penalty


    n_features=len(new_row)
    print(f"data has {n_features} features")
    denoiser=Unet1DModelCLassy(in_channels=n_features,out_channels=n_features,
        use_timestep_embedding=True,use_class_embedding=True,num_classes=len(config_class_set))
    model_parameters=[p for p in denoiser.parameters()]
    print(f"optimzing {len(model_parameters)} model params")
    optimizer=torch.optim.AdamW(model_parameters)
    scheduler=DDIMScheduler(num_train_timesteps=args.timesteps)

    optimizer,scheduler,denoiser=accelerator.prepare(optimizer,scheduler,denoiser)
    input_data=batchify(input_data,args.batch_size)
    config_class_data=batchify(config_class_data)

    for e in range(args.epochs):
        start=time.time()
        loss_list=[]
        for b in range(args.batches_per_epoch):
            input_batch=input_data[b]
            class_labels=config_class_data[b]
            optimizer.zero_grad()
            noise=torch.randn((args.batch_size,n_features))

            timesteps = torch.randint(0, args.timesteps, (args.batch_size,)).long()

            noisy_inputs=scheduler.add_noise(input_batch,noise,timesteps)
            
            

            with accelerator.accumulate(denoiser):
                # Predict the noise residual
                noise_pred=denoiser(noisy_inputs,timesteps,class_labels,return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(denoiser.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item()}
            loss_list.append(loss.detach().item())
            accelerator.log(logs)
        end=time.time()
        avg_loss=np.mean(loss_list)
        print(f"epoch {e} elapsed {end-start} seconds avg loss {avg_loss} ")
        accelerator.log({"average loss":avg_loss})

    for s in range(args.optimization_samples):
        vector=torch.randn((1,n_features))
        random_index = torch.randint(0, len(config_class_set), (1,)).item()

        # Create a one-hot vector
        class_labels = torch.zeros(len(config_class_set))
        class_labels[random_index] = 1
        class_labels=class_labels.unsqueeze(0)
        start=time.time()
        with accelerator.accumulate(denoiser):
            optimizer.zero_grad()
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler, args.inference_steps,
            )
            for i,t in enumerate(timesteps):
                vector_input=scheduler.scale_model_input(vector,t)

                noise_pred=denoiser(vector_input,t,class_labels,return_dict=False)[0]

                vector = scheduler.step(noise_pred, t, vector, return_dict=False)[0]

            loss=calculate_penalty(vector,args.minimize)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        end=time.time()
        print(f" sample {s} elpased {end-start} seconds loss = {loss.detach().item()}")


        

if __name__=="__main__":
    args=parser.parse_args()
    main(args)