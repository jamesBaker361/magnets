import gymnasium as gym
import torch
from gymnasium import spaces
from modeling import get_model
import numpy as np
from scipy.optimize import minimize_scalar
from simulation import get_function

class MagneticOptimizationEnv(gym.Env):
    def __init__(self,target_values:dict,state_dict_path:str,max_steps:int=500,radius_min:float=0.1):
        super(MagneticOptimizationEnv,self).__init__()

        self.upper_limits=[0.5 for _ in range(6)]+[100 for _ in range(6)]+[0.01 for _ in range(6)]

        self.radius_min=radius_min

        self.action_space=spaces.Box(low=[0.0 for _ in range(18)],high=self.upper_limits)

        self.target_values=target_values

        self.observation_space=spaces.Box(low=[-10000]+[0.0 for _ in range(18)],high=[0]+self.upper_limits)

        self.model=get_model(21)

        self.model.load_state_dict(torch.load(state_dict_path))
        self.step_count=0
        self.max_steps=max_steps

    def calculate_loss(self,params:list):
        loss=0.0
        for [x,y,z,b_x,b_y,b_z] in self.target_values:
            predicted=self.model(torch.tensor([x,y,z]+params))
            loss-=np.linalg.norm(np.array([b_x,b_y,b_z]) - np.array(predicted))

        radius_function=get_function()
        return loss
    
    def step(self,action):

        action=np.clip(action, [0 for _ in range(18)], self.upper_limits)
        reward=self.calculate_loss(action)

        self.step_count+=1
        if self.step_count>=self.max_steps or reward>=0.000001:
            terminated=True
        else:
            terminated=False
        
        
        info={}
        truncated=False

        observation=[reward]
        return observation, reward, terminated, truncated, info
