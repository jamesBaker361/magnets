import torch
import argparse
from torch.nn import Linear,Dropout,LeakyReLU,Sequential
import os
import numpy as np
import json
import csv
from accelerate import Accelerator

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
    output_data=[]
    error_data=[]
    propellant_class_set=set()
    A_mat_class_set=set()
    C_mat_class_set=set()
    config_class_set=set()
    with open(args.csv_file,"r",encoding="cp1252") as file:
        dict_reader=csv.DictReader(file)
        for d_row in dict_reader:
            propellant_class_set.add(d_row["propellant"])
            A_mat_class_set.add(d_row["A_mat"])
            C_mat_class_set.add(d_row["C_mat"])
            config_class_set.add(d_row["config"])
        propellant_dict=set_to_one_hot(propellant_class_set)
        A_mat_dict=set_to_one_hot(A_mat_class_set)
        C_mat_dict=set_to_one_hot(C_mat_class_set)
        config_dict=set_to_one_hot(config_class_set)
    with open(args.csv_file,"r",encoding="cp1252") as file:
        reader = csv.reader(file)
        for row in reader:
            quantitative=[float_handle_na(d) for d in row[:14]]
            T_tot,J,B_A,mdot,error,Ra,Rc,Ra0,La,Rbi,Rbo,Lc_a,V,Pb=quantitative
            qualitative=row[14:-1]
            propellant,source,thruster,A_mat,C_mat,config=qualitative
            new_row=[J,B_A,Ra,Rc,Ra0,La,Rbi,Rbo,Lc_a,V]
            new_row=np.concatenate((new_row, propellant_dict[propellant], A_mat_dict[A_mat], C_mat_dict[C_mat], config_dict[config]))
            input_data.append(new_row)
            output_data.append(T_tot)
            error_data.append(error)
    

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