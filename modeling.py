import torch
import os
import argparse
from torch import nn
import torch.optim as optim
import time
import numpy as np
import wandb

parser=argparse.ArgumentParser()
parser.add_argument("--limit",type=int,default=100)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--validation",action="store_true")
parser.add_argument("--project_name",type=str,default="magnetism")

def get_model(n_inputs:int)-> torch.nn.Sequential:
    return torch.nn.Sequential(*[
        torch.nn.Linear(n_inputs,100),
        torch.nn.LeakyReLU(),
        nn.Dropout(),
        torch.nn.Linear(100,50),
        torch.nn.LeakyReLU(),
        nn.Dropout(),
        torch.nn.Linear(50,10),
        nn.LeakyReLU(),
        nn.Linear(10,3)
    ])


if __name__=="__main__":
    args=parser.parse_args()

    #load data
    data_folder="fake_simulation_data"
    os.makedirs(data_folder,exist_ok=True)

    wandb.init(
    project=args.project_name,  # Change to your project name
    config=vars(args)
    )

    rows=[]
    for  f in os.listdir(data_folder):
        file=open(os.path.join(data_folder,f))
        first_line=file.readline()
        for line in file:
            line_list=list(map(float,line.split(",")))
            elapsed,L,x,y,z,b_x,b_y,b_z=line_list[:8]
            params=line_list[8:]
            row=[b_x,b_y,b_z,x,y,z]+params
            rows.append(row)

    rows=rows[:args.limit]
    n_inputs=len(row)-3

    model=get_model(n_inputs)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    split=int(0.7*len(rows))
    test_split=int(0.8*len(rows))
    train_data=rows[:split]
    train_data=[torch.tensor(train_data[t:t+args.batch_size]) for t in range(0,len(train_data), args.batch_size)]

    val_data=rows[split:test_split]
    val_data=[torch.tensor(val_data[t:t+args.batch_size]) for t in range(0,len(val_data), args.batch_size)]
    test_data=rows[test_split:]
    test_data=[torch.tensor(test_data[t:t+args.batch_size]) for t in range(0,len(test_data), args.batch_size)]

    for e in range(args.epochs):
        loss_list=[]
        start=time.time()
        for batch in train_data:
            
            output_data,input_data=torch.split(batch,[3,n_inputs],dim=1)
            predicted=model(input_data)

            loss=criterion(predicted,output_data)

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()        # Backpropagation
            optimizer.step()       # Update weights
            loss_list.append(loss.detach()/args.batch_size)
        train_loss=np.mean(loss_list)
        print(f"epoch {e} ended after {time.time()-start} seconds w avg loss {train_loss}")
        wandb.log({"train_loss":train_loss})
        if args.validation:
            val_loss_list=[]

            start=time.time()
            with torch.no_grad():
                for batch in val_data:
                    output_data,input_data=torch.split(batch,[3,n_inputs],dim=1)
                    predicted=model(input_data)

                    loss=criterion(predicted,output_data)

                    val_loss_list.append(loss/args.batch_size)
                val_loss=np.mean(val_loss_list)
                print(f"validation epoch {e} ended after {time.time()-start} seconds w avg loss {val_loss}")
                wandb.log({"val_loss":val_loss})
        
    test_loss_list=[]
    with torch.no_grad():
        for batch in test_data:
            output_data,input_data=torch.split(batch,[3,n_inputs],dim=1)
            predicted=model(input_data)

            loss=criterion(predicted,output_data)
            test_loss_list.append(loss/args.batch_size)

    test_loss=np.mean(test_loss_list)
    print(f"test loss {test_loss}")
    wandb.log({"test_loss":test_loss})