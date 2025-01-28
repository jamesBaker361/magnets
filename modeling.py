import torch
import argparse
from torch.nn import Linear,Dropout,LeakyReLU,Sequential
import os
import numpy as np
import json

parser=argparse.ArgumentParser()
parser.add_argument("--src_dir_list",nargs="*")
parser.add_argument("--n_coils",type=int,default=4)
parser.add_argument("--max_fourier_mode",type=int,default=2)
parser.add_argument("--nozzle_radius",type=float,default=0.1,help="nozzle radius for velocity")
parser.add_argument("--radius",type=float,default=1.0,help="chamber radius- maximum for particles, minimum for coils")
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--test_fraction",type=float,default=0.1)
parser.add_argument("--epochs",type=int,default=50)
parser.add_argument("--n_layers",type=int,default=3)
parser.add_argument("--name",type=str,default="model")

SAVE_MODEL_PATH="/scratch/jlb638/magnet_models"
os.makedirs(SAVE_MODEL_PATH,exist_ok=True)

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
    input_data=[]
    output_data=[]
    files_used=0
    for directory in args.src_dir_list:
        csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
        for file_path in csv_files:
            with open(os.path.join(directory,file_path),"r") as csv_file:
                first_line=csv_file.readline().strip()
                [total_coefficients,n_coils,radius,max_fourier_mode,objective,nozzle_radius,n_particles]=first_line.split(",")
                total_coefficients=int(total_coefficients)
                n_coils=int(n_coils)
                max_fourier_mode=int(max_fourier_mode)
                if args.n_coils==n_coils and args.max_fourier_mode==max_fourier_mode:
                    files_used+=1
                    for line in csv_file:
                        row=line.strip().split(",")
                        reward=row[0]
                        observation=[float(o) for o in row[1:8]]
                        vector=[float(v) for v in row[8:11]]
                        velocity=[float(row[11])]
                        coefficients=[float(c) for c in row[12:12+total_coefficients]]
                        assert len(coefficients)==total_coefficients
                        amps=[float(a) for a in row[12+total_coefficients:]]
                        assert len(amps)==args.n_coils
                        input_data.append(np.concatenate([vector,velocity,coefficients,amps]))
                        output_data.append(observation)
    
    print(f"used {files_used} files")
    print(f"found {len(input_data)} rows of data")
    input_data_batched=[torch.tensor(input_data[i:i+args.batch_size],dtype=torch.float32) for i in range(0,len(input_data),args.batch_size)]
    output_data_batched=[torch.tensor(output_data[o:o+args.batch_size],dtype=torch.float32) for o in range(0,len(output_data),args.batch_size)]
    test_limit=int(0.1*len(input_data_batched))
    test_input_data_batched=input_data_batched[:test_limit]
    input_data_batched=input_data_batched[test_limit:]
    test_output_data_batched=output_data_batched[:test_limit]
    output_data_batched=output_data_batched[test_limit:]

    input_dim=4+total_coefficients+n_coils
    sim_model=SimulationModel(input_dim,7,args.n_layers).to(torch.float32)

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
    

    torch.save(sim_model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{args.name}.pth"))
    config={"input_dim":input_dim, "output_dim":7,"n_layers":args.n_layers}

    with open(os.path.join(SAVE_MODEL_PATH,f"{args.name}.json"), "w") as json_file:
        json.dump(config, json_file, indent=4)

    test_loss=[]
    for input_batch,output_batch in zip(test_input_data_batched,test_output_data_batched):
        predicted=sim_model(input_batch)
        loss=criterion(predicted,output_batch)
        test_loss.append(loss.detach())
    print(f"test loss: {np.average(test_loss)}")

        

if __name__=="__main__":
    args=parser.parse_args()
    training_loop(args)