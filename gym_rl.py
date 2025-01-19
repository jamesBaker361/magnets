import gymnasium as gym
import torch
from gymnasium import spaces
from modeling import get_model
import numpy as np
from scipy.optimize import minimize_scalar
from simulation import get_function
from stable_baselines3 import PPO
import time

class MagneticOptimizationEnv(gym.Env):
    def __init__(self,target_values:dict,state_dict_path:str,radius_min:float=0.1):
        super(MagneticOptimizationEnv,self).__init__()

        self.upper_limits=np.array([0.5 for _ in range(6)]+[100 for _ in range(6)]+[0.01 for _ in range(6)])

        self.radius_min=radius_min

        self.action_space=spaces.Box(low=np.array([0.0 for _ in range(18)]),high=self.upper_limits)

        self.target_values=target_values

        self.observation_space=spaces.Box(low=np.array([-10000]+[0.0 for _ in range(18)]),high=np.concatenate(np.array([0.0]),self.upper_limits))

        self.model=get_model(21)

        self.model.load_state_dict(torch.load(state_dict_path))


    def calculate_loss(self,params:list):
        loss=0.0
        for [x,y,z,b_x,b_y,b_z] in self.target_values:
            predicted=self.model(torch.tensor([x,y,z]+params))
            loss-=np.linalg.norm(np.array([b_x,b_y,b_z]) - np.array(predicted))

        radius_function=get_function(params[:6])
        test_radius_minimum = minimize_scalar(radius_function, bounds=(0, 1), method='bounded')
        if test_radius_minimum<self.radius_min:
            return -10000

        n_turn_function=get_function(params[6:12])
        test_n_turns=minimize_scalar(n_turn_function,bounds=(0, 1), method='bounded')
        if test_n_turns<0:
            return -10000
        
        thickness_function=get_function(params[12:])
        test_thickness=minimize_scalar(thickness_function, bounds=(0,1), method="bounded")
        if test_thickness<0:
            return -10000
        return loss
    
    def step(self,action):

        action=np.clip(action, [0 for _ in range(18)], self.upper_limits)
        reward=self.calculate_loss(action)

        if reward>=-0.00001:
            terminated=True
        else:
            terminated=False
        
        
        info={}
        truncated=False

        observation=[reward]
        return observation, reward, terminated, truncated, info


if __name__=="__main__":
    target_values=[[0.5,0.5,float(z)/10, 0,0,1] for z in range(10)] 
    env=MagneticOptimizationEnv(target_values,"/scratch/jlb638/magnet_model/test_model.pth")
    # Train PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    start=time.time()
    model.learn(total_timesteps=500)
    print(f"elapsed {time.time()-start} seconds")