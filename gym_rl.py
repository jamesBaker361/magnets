import gymnasium as gym
import torch
from gymnasium import spaces
from modeling import get_model

class MagneticOptimizationEnv(gym.Env):
    def __init__(self,target_values:dict,state_dict_path:str):
        super(MagneticOptimizationEnv,self).__init__()

        upper_limits=[0.5 for _ in range(6)]+[100 for _ in range(6)]+[0.01 for _ in range(6)]

        self.action_space=spaces.Box(low=[0.0 for _ in range(18)],high=upper_limits)

        self.target_values=target_values

        self.observation_space=spaces.Box(low=[0.0 for _ in range(19)],high=[1000]+upper_limits)

        self.model=get_model(21)

        self.model.load_state_dict(torch.load(state_dict_path))

    def calculate_loss(self,params:list):
        for [x,y,z,b_x,b_y,b_z] in self.target_values:
            predicted=self.model(torch.tensor([x,y,z]+params))