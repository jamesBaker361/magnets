import gymnasium as gym
import torch
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

class MagneticOptimizationEnv(gym.Env):
    def __init__(self,start_positions:list,start_velocities:list,n_coils:int,max_fourier_n:int,nozzle_radius:float,radius:float,n_particles:int):
        super(MagneticOptimizationEnv,self).__init__()
        self.start_positions=start_positions #places where we might start a particle
        self.start_velocities=start_velocities
        self.n_coils=n_coils #how many coils
        self.max_fourier_n=max_fourier_n
        self.nozzle_radius=nozzle_radius
        self.radius=radius
        self.n_particles=n_particles

        self.parameters_per_coil=2+(2*max_fourier_n)+2 #constant x,y + cos,sin for each mode x,y + z + current
        upper_limits_per_coil=[1.0]+[1.0 for _ in range(2*max_fourier_n)] +[1.0]+[10000]
        upper_limits=np.concatenate([upper_limits_per_coil for _ in range(n_coils)])
        lower_limits_per_coil=[0.]+[0.0 for _ in range(2*max_fourier_n)]+[0.]+[100]
        lower_limits=np.concatenate([lower_limits_per_coil for _ in range(n_coils)])

        
        #self.observation_space=spaces.Box(low=lower_limits,high=upper_limits)
        self.action_space=spaces.Box(low=lower_limits,high=upper_limits)
        particles_upper_limits=[1 for _ in range(3)]+[10000 for _ in range(3)] #distance and velocity vectors
        particles_lower_limits=[-1 for _ in range(3)]+[-10000 for _ in range(3)]
        self.observation_space=spaces.Box(low=np.concatenate([particles_lower_limits for _ in range(n_particles)]),
            high=np.concatenate([particles_upper_limits for _ in range(n_particles)])
        )


    def calculate_reward(self,observation:list):
        reward=0.0
        for [x,y,z,v_x,v_y,v_z] in observation:
            if np.linalg.norm([x,y])< self.nozzle_radius:
                reward+=v_z #for each particle, that is in the nozzle, we want as much z momentumas possible
    
    def step(self,action):

        #action=np.clip(action, [0 for _ in range(18)], self.upper_limits)
        #calculate particle locations
        coil_list=[]
        for c in range(self.n_coils):
            coil_parameters=action[c:c+self.parameters_per_coil]
            curve = CurveXYZFourier(1000, self.max_fourier_n)  # 100 = Number of quadrature points, 1 = max Fourier mode number
            fourier_amplitudes=coil_parameters[:-2]
            curve.x = fourier_amplitudes+ coil_parameters[-2]+[0. for _ in range(2*self.max_fourier_n)]  # Set Fourier amplitudes
            coil = Coil(curve, Current(coil_parameters[-1]))  # 10 kAmpere-turns
            coil_list.append(coil)

        field=BiotSavart([coil_list])
        field=InterpolatedField(field)
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        res_tys,res_phi_hits=trace_particles(field,self.start_positions,self.start_velocities,mass=m,charge=q)

        
        return #observation, reward, terminated, truncated, info


if __name__=="__main__":
    target_values=[[0.5,0.5,float(z)/10, 0,0,1] for z in range(10)] 
    env=MagneticOptimizationEnv(target_values,"/scratch/jlb638/magnet_model/test_model.pth")
    # Train PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    start=time.time()
    model.learn(total_timesteps=500)
    print(f"elapsed {time.time()-start} seconds")