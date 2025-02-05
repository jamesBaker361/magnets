import torch
import argparse
from torch.nn import Linear,Dropout,LeakyReLU,Sequential
import os
import numpy as np
import json
import csv
from accelerate import Accelerator
import random
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--test_fraction",type=float,default=0.1)
parser.add_argument("--epochs",type=int,default=50)
parser.add_argument("--n_layers",type=int,default=3)
parser.add_argument("--name",type=str,default="model")
parser.add_argument("--csv_file",type=str,default="AFMPDT_database.csv")
parser.add_argument("--gradient_accumulation_steps",type=int,default=2)
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="thrust_modeling")

def batchify(input_data,batch_size):
    input_data_batched=[torch.tensor(input_data[i:i+batch_size],dtype=torch.float32) for i in range(0,len(input_data),batch_size)]
    return input_data_batched

SAVE_MODEL_PATH="/scratch/jlb638/magnet_models"
os.makedirs(SAVE_MODEL_PATH,exist_ok=True)

def set_to_one_hot(input_set):
    """
    Convert a set of unique items into a dictionary of one-hot vectors.
    
    Args:
        input_set (set): A set of unique items.
    
    Returns:
        dict: A dictionary where keys are the items from the set and values are one-hot vectors.
    """
    # Convert the set into a sorted list to ensure consistent order
    items = sorted(input_set)
    
    # Create a dictionary mapping each item to its one-hot vector
    one_hot_dict = {
        item: [1 if i == idx else 0 for i in range(len(items))]
        for idx, item in enumerate(items)
    }
    
    return one_hot_dict

def float_handle_na(d):
    try:
        return float(d)
    except:
        return 0.

class SimulationModel(torch.nn.Module):
    def __init__(self,input_dim:int,final_output_dim:int,n_layers:int):
        super().__init__()
        layer_list=[]
        layer_step=int(float(input_dim-final_output_dim)/n_layers)
        for l in range(n_layers-1):
            output_dim=input_dim-layer_step
            layer_list.append(Linear(input_dim,output_dim))
            layer_list.append(LeakyReLU())
            layer_list.append(Dropout(0.1))
            input_dim=output_dim
        layer_list.append(Linear(input_dim,final_output_dim))
        self.model=Sequential(*layer_list)

    def forward(self,x):
        return self.model(x)
    
def training_loop(args):

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
    cols_to_normalize =["T_tot","J", "mdot", "B_A", "Ra", "Rc", "Ra0", "La", "Rbi", "Rbo", "Lc_a", "V", "Pb"]

    # Compute mean and standard deviation for each column
    means = df[cols_to_normalize].mean()
    stds = df[cols_to_normalize].std()  # Standard deviation

    # Normalize the columns using Z-score normalization: (x - mean) / std
    df[cols_to_normalize] = (df[cols_to_normalize] - means) / stds

    

    df["quant"]=df.apply(lambda row: np.concatenate((
        row[["J", "mdot","B_A", "Ra", "Rc", "Ra0", "La", "Rbi", "Rbo", "Lc_a", "V", "Pb"]].values,
    )), axis=1)

    cols_to_normalize=cols_to_normalize[1:]
    

    # Apply transformation and store in "complete"
    df["complete"] = df.apply(lambda row: np.concatenate((
        row[cols_to_normalize].values,  # Already normalized
        propellant_dict[row["propellant"]],
        A_mat_dict[row["A_mat"]],
        C_mat_dict[row["C_mat"]],
        config_dict[row["config"]]  # Wrap in a list to keep shape
    )), axis=1)

    # Prepare input data and labels
    input_data = df["complete"].tolist()
    new_row=input_data[0]
    output_data=df["T_tot"].tolist()
    

    print(f"found {len(input_data)} rows of data")

    input_data_batched=batchify(input_data,args.batch_size)
    output_data_batched=batchify(output_data,args.batch_size)
    test_limit=int(0.1*len(input_data_batched))
    test_input_data_batched=input_data_batched[:test_limit]
    input_data_batched=input_data_batched[test_limit:]
    test_output_data_batched=output_data_batched[:test_limit]
    output_data_batched=output_data_batched[test_limit:]

    input_dim=len(new_row)
    
    sim_model=SimulationModel(input_dim,1,args.n_layers).to(torch.float32)

    optimizer=torch.optim.AdamW([p for p in sim_model.parameters()])
    criterion=torch.nn.MSELoss(reduction='mean')

    total_loss_list=[]
    for e in range(args.epochs):
        loss_list=[]
        for input_batch,output_batch in zip(input_data_batched,output_data_batched):
            optimizer.zero_grad()
            predicted=sim_model(input_batch)
            loss=criterion(predicted,output_batch)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.detach())
        print(f" epoch {e} avg loss {np.average(loss_list)}")
        accelerator.log({"avg_loss":np.average(loss_list)})
    

    torch.save(sim_model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{args.name}.pth"))
    config={"input_dim":input_dim, "output_dim":1,"n_layers":args.n_layers}

    with open(os.path.join(SAVE_MODEL_PATH,f"{args.name}.json"), "w") as json_file:
        json.dump(config, json_file, indent=4)

    test_loss=[]
    for input_batch,output_batch in zip(test_input_data_batched,test_output_data_batched):
        predicted=sim_model(input_batch)
        loss=criterion(predicted,output_batch)
        test_loss.append(loss.detach())
    print(f"test loss: {np.average(test_loss)}")
    accelerator.log({"test_loss":np.average(test_loss)})

        

if __name__=="__main__":
    args=parser.parse_args()
    training_loop(args)