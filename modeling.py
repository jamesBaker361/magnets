import torch
import argparse
from torch.nn import Linear,Dropout,LeakyReLU,Sequential

parser=argparse.ArgumentParser()

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
        

if __name__=="__main__":
    batch_size=2
    for input_dim in [12,24]:
        for n_layers in [2,3,4]:
            sim_model=SimulationModel(input_dim,7,n_layers)
            noise=torch.randn((batch_size,input_dim))
            sim_model(noise)
            print(input_dim,n_layers)
    args=parser.parse_args()