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
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from torch.nn import Linear,Sequential
import gymnasium as gym
import numpy as np
import csv
from modeling import set_to_one_hot,float_handle_na
from diffusers import DDIMScheduler
from diffusers.models.embeddings import TimestepEmbedding,LabelEmbedding
from unet_1D_with_class import Unet1DModelCLassy
import random
from modeling import batchify,SimulationModel,SAVE_MODEL_PATH
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.models.unets.unet_1d_blocks import get_mid_block, get_out_block, get_up_block,Downsample1d,ResConvBlock,SelfAttention1d,DownBlockType,DownBlock1D,DownBlock1DNoSkip,DownResnetBlock1D
from torch.nn import ModuleList,Sequential
import os
import pandas as pd
import sys


AMPS=1000
m = PROTON_MASS
q = ELEMENTARY_CHARGE
Ekin = 10*ONE_EV
PRESSURE=1 #pressure in low earth orbit

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--csv_file",type=str,default="AFMPDT_database.csv")
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--n_layers",type=int,default=4)
parser.add_argument("--n_layers_simulation_model",type=int,default=3)
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
parser.add_argument("--simulation_model_name",type=str,default="model_3_100")
parser.add_argument("--no_training_optimization",action="store_true")
parser.add_argument("--conditional",action="store_true")
parser.add_argument("--sample_optimization",action="store_true")
parser.add_argument("--evaluation_samples",type=int,default=20)
#parser.add_argument("--pretraining_penalty",action="store_true",help="whether to do the penalty during pretraining")



class Denoiser(torch.nn.Module):
    def __init__(self, n_features:int, n_layers:int,residuals:bool,increasing:bool,
                 num_classes:int,
                 time_embedding_type:str= "fourier",
                 embedding_size:int=32, #size of class + time embeddding
                 use_class_embedding:bool=True,
                 flip_sin_to_cos:bool=True,
                 act_fn = "swish",
                 freq_shift:int=0,
                 hidden_state_size:int=32):
        super().__init__()
        self.n_features=n_features
        self.n_layers=n_layers
        self.residuals =residuals
        diff=n_features/n_layers
        down_layer_list=[]
        up_layer_list=[]
        prev=hidden_state_size
        for n in range(n_layers//2):
            if increasing:
                current=prev+diff
            else:
                current=prev-diff
            
            down_layer_list.append(Sequential(  Linear(int(prev),int(current))))
            print("down layer",prev,current)
            prev=current
        
        '''if increasing:
            self.mid_block=Linear(int(prev),int(hidden_state_size*2))
            prev=hidden_state_size*2
        else:
            self.mid_block=Linear(int(prev),int(hidden_state_size//2))
            prev=hidden_state_size//2'''
        
        for n in range(n_layers//2):
            if increasing:
                current=prev-diff
            else:
                current=prev+diff
            up_layer_list.append(Sequential( Linear(int(prev),int(current))))
            print("up layer ",prev,current)
            prev=current
        self.down_block=down_layer_list
        self.up_block=up_layer_list
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=embedding_size, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = 2 * embedding_size
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
               embedding_size, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift
            )
            timestep_input_dim = embedding_size
        self.use_class_embedding=use_class_embedding
        time_embed_dim = embedding_size * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=timestep_input_dim,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=embedding_size//2,
        )
        self.class_embedding=torch.nn.Embedding(num_classes,embedding_size//2)

        self.linear_in=Linear(n_features+embedding_size,hidden_state_size)
        self.linear_out=Sequential(  Linear(hidden_state_size,n_features))
    def forward(self,x,time_step, class_label):
        time_emb=self.time_mlp(self.time_proj(time_step))
        class_emb=self.class_embedding(class_label)
        x=torch.cat([x,class_emb,time_emb],dim=-1)

        x=self.linear_in(x)

        if self.residuals:
            residual_list=[]
            for linear in self.down_block:
                residual_list.append(x)
                x=linear(x)
                
            #x=self.mid_block(x)
            residual_list=residual_list[::-1]
            for i,linear in enumerate(self.up_block):
                
                x=linear(x)
                x=x+residual_list[i]
        else:
            x=self.down_block(x)
            x=self.mid_block(x)
            x=self.up_block(x)
        x=self.linear_out(x)
        return x
    
#TODO: binary/TF classifier for evaluation
#diffusion conditioning
#less class embedding stuff


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
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(args.csv_file, encoding="cp1252")
    df = df.fillna("NaN") 

    # Extract unique class sets for categorical variables
    propellant_class_set = set(df["propellant"])
    A_mat_class_set = set(df["A_mat"])
    print(A_mat_class_set)
    C_mat_class_set = set(df["C_mat"])
    config_class_set = set(df["config"])

    # Convert class sets to one-hot encoding dictionaries
    propellant_dict = set_to_one_hot(propellant_class_set)
    A_mat_dict = set_to_one_hot(A_mat_class_set)
    C_mat_dict = set_to_one_hot(C_mat_class_set)
    config_dict=set_to_one_hot(config_class_set)

    # Map config classes to integer labels
    config_int_dict = {label: index for index, label in enumerate(config_class_set)}

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)


    # Process quantitative data (first 14 columns)
    quantitative_columns = df.columns[:14]
    df[quantitative_columns] = df[quantitative_columns].applymap(float_handle_na)
    n_quantitative_inputs=12
    # Extract qualitative columns
    df["new_row"] = df.apply(lambda row: np.concatenate((
        row[["J", "mdot","B_A", "Ra", "Rc", "Ra0", "La", "Rbi", "Rbo", "Lc_a", "V", "Pb"]].values,
        propellant_dict[row["propellant"]],
        A_mat_dict[row["A_mat"]],
        C_mat_dict[row["C_mat"]]
    )), axis=1)

    

    df["quant"]=df.apply(lambda row: np.concatenate((
        row[["J", "mdot","B_A", "Ra", "Rc", "Ra0", "La", "Rbi", "Rbo", "Lc_a", "V", "Pb"]].values,
    )), axis=1)

    cols_to_normalize = ["J", "mdot", "B_A", "Ra", "Rc", "Ra0", "La", "Rbi", "Rbo", "Lc_a", "V", "Pb"]

    # Compute mean and standard deviation for each column
    means = df[cols_to_normalize].mean()
    stds = df[cols_to_normalize].std()  # Standard deviation

    # Normalize the columns using Z-score normalization: (x - mean) / std
    df[cols_to_normalize] = (df[cols_to_normalize] - means) / stds

    # Apply transformation and store in "complete"
    df["complete"] = df.apply(lambda row: np.concatenate((
        row[cols_to_normalize].values,  # Already normalized
        propellant_dict[row["propellant"]],
        A_mat_dict[row["A_mat"]],
        C_mat_dict[row["C_mat"]],
        config_dict[row["config"]]  # Wrap in a list to keep shape
    )), axis=1)

    # Prepare input data and labels
    input_data = df["new_row"].tolist()
    new_row=input_data[0]
    config_class_data = df["config"].map(config_int_dict).tolist()
    complete_data=df["complete"].to_list()
    quant_data=df["quant"].to_list()

    #means = np.mean(quant_data, axis=0)
    #stds = np.std(quant_data, axis=0)
    print(complete_data[0])
    print(quant_data[0])
    print("n propellant",len(propellant_dict))
    print("len A_mat", len(A_mat_dict))
    print("len c mat",len(C_mat_dict))

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
        

    def straight_through_one_hot(x):
        """
        Differentiable hard one-hot encoding using Straight-Through Estimator (STE).
        
        Args:
            x (Tensor): Input tensor of shape (batch, n).
            
        Returns:
            Tensor: Hard one-hot encoded tensor with gradients.
        """
        soft = F.softmax(x, dim=-1)  # Softmax for gradients
        hard = torch.argmax(soft, dim=-1)  # Hard decision
        one_hot = F.one_hot(hard, num_classes=x.shape[-1]).float()  # One-hot encoding
        return (one_hot - soft).detach() + soft  # STE trick
    
    def denoise_predictions_to_simulation_input(sample,class_label):
        config=torch.tensor(config_dict[class_label]).unsqueeze(0)
        quant_inputs=sample[:n_quantitative_inputs]
        A_mat_start=n_quantitative_inputs+len(propellant_dict)
        C_mat_start=A_mat_start+len(A_mat_dict)
        quant_inputs,propellant,A_mat,C_mat=torch.split(sample,[n_quantitative_inputs, len(propellant_dict), len(A_mat_dict),len(C_mat_dict)],dim=1)
        #print(n_quantitative_inputs, A_mat_start,C_mat_start,len(sample))
        propellant=straight_through_one_hot(propellant)
        A_mat=straight_through_one_hot(A_mat)
        C_mat=straight_through_one_hot(C_mat)
        return torch.cat([quant_inputs, propellant,A_mat,C_mat,config],dim=-1)

    simulation_inputs=len(new_row)+len(config_int_dict)
    simulation_model=SimulationModel(simulation_inputs,1, args.n_layers_simulation_model)

    simulation_path=os.path.join(SAVE_MODEL_PATH, f"{args.simulation_model_name}.pth")

    #simulation_model.load_state_dict(torch.load(simulation_path))

    def find_nearest_training_sample(sample,class_label):
        #sample[:, :n_quantitative_inputs] = (sample[:, :n_quantitative_inputs] - means) / stds 
        vector_sample=denoise_predictions_to_simulation_input(sample,class_label).numpy()
        
        vals=[]
        for s in vector_sample:
            for n in range(n_quantitative_inputs):
                s[n]=(s[n]-means[n])/stds[n]
            min_sim=sys.maxsize
            for row in complete_data:
                min_sim=min(min_sim,np.linalg.norm(s-row))
        return vals

        

        

    sm_penalty=SquaredMagnitudePenalty()
    def calculate_penalty(sample:torch.Tensor,class_label,minimize):
        simulation_tensor=denoise_predictions_to_simulation_input(sample,class_label)
        thrust=-simulation_model(  simulation_tensor)
        if minimize:
            thrust+=sm_penalty(sample)
        return thrust


    n_features=len(new_row)
    print(f"data has {n_features} features")
    denoiser=Denoiser(n_features,4,True,True,len(config_class_set))
    
    model_parameters=[p for p in denoiser.parameters()]
    print(f"optimzing {len(model_parameters)} model params")
    optimizer=torch.optim.AdamW(model_parameters)
    scheduler=DDIMScheduler(num_train_timesteps=args.timesteps)

    optimizer,scheduler,denoiser=accelerator.prepare(optimizer,scheduler,denoiser)
    input_data_batched=batchify(input_data,args.batch_size)
    config_class_data_batched=batchify(config_class_data,args.batch_size)

    for e in range(args.epochs):
        start=time.time()
        loss_list=[]
        for b in range(args.batches_per_epoch):
            input_batch=input_data_batched[b]
            class_labels=config_class_data_batched[b].long()
            optimizer.zero_grad()
            noise=torch.randn((args.batch_size,n_features))

            timesteps = torch.randint(0, args.timesteps, (args.batch_size,)).long()

            noisy_inputs=scheduler.add_noise(input_batch,noise,timesteps)

            with accelerator.accumulate(denoiser):
                # Predict the noise residual
                noise_pred=denoiser(noisy_inputs,timesteps,class_labels)
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
        random_class_label = random.choice(list(config_class_set))
        class_labels =torch.tensor(config_int_dict[random_class_label]).unsqueeze(0).long()
        start=time.time()
        with accelerator.accumulate(denoiser):
            optimizer.zero_grad()
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler, args.inference_steps,
            )
            for i,t in enumerate(timesteps):
                t=t.unsqueeze(0).long()
                vector_input=scheduler.scale_model_input(vector,t)

                noise_pred=denoiser(vector_input,t,class_labels)

                vector = scheduler.step(noise_pred, t, vector, return_dict=False)[0]

            loss=calculate_penalty(vector,minimize=args.minimize,class_label=random_class_label)
            accelerator.backward(loss)
            find_nearest_training_sample(vector.detach(),random_class_label)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        end=time.time()
        print(f" sample {s} elpased {end-start} seconds loss = {loss.detach().item()}")

    generated_samples=[]
    for _ in range(args.evaluation_samples):
        vector=torch.randn((1,n_features))
        random_index = torch.randint(0, len(config_class_set), (1,)).item()

        # Create a one-hot vector
        random_class_label = random.choice(list(config_class_set))
        class_labels =torch.tensor(config_int_dict[random_class_label]).unsqueeze(0).long()
        start=time.time()
        with accelerator.accumulate(denoiser):
            optimizer.zero_grad()
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler, args.inference_steps,
            )
            for i,t in enumerate(timesteps):
                t=t.unsqueeze(0).long()
                vector_input=scheduler.scale_model_input(vector,t)

                noise_pred=denoiser(vector_input,t,class_labels)

                vector = scheduler.step(noise_pred, t, vector, return_dict=False)[0]

            
            generated_samples.append(vector)

        

if __name__=="__main__":
    args=parser.parse_args()
    main(args)