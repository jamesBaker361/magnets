print("began importing")
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
from simsopt.field.tracing import MinZStoppingCriterion, MaxRStoppingCriterion,MaxZStoppingCriterion
print("done importing!")

class MagneticOptimizationEnv(gym.Env):
    def __init__(self,start_positions:list,start_velocities:list,n_coils:int,max_fourier_n:int,nozzle_radius:float,radius:float,regularization_lambda:float):
        super(MagneticOptimizationEnv,self).__init__()
        print("began constucting")
        start_positions=np.array(start_positions)
        self.start_positions=start_positions #places where we might start a particle
        self.start_velocities=start_velocities
        self.n_coils=n_coils #how many coils
        self.max_fourier_n=max_fourier_n
        self.nozzle_radius=nozzle_radius
        self.radius=radius
        n_particles=len(start_positions)
        self.n_particles=n_particles
        self.regularization_lambda=regularization_lambda

        self.parameters_per_coil=2+(4*max_fourier_n)+2 #constant x,y + cos,sin for each mode x,y + z + current
        upper_limits_per_coil=[1.0,1.0]+[1.0 for _ in range(4*max_fourier_n)] +[1.0]+[10000]
        upper_limits=np.concatenate([upper_limits_per_coil for _ in range(n_coils)])
        lower_limits_per_coil=[0.,0.]+[0.0 for _ in range(4*max_fourier_n)]+[0.]+[100]
        lower_limits=np.concatenate([lower_limits_per_coil for _ in range(n_coils)])

        
        #self.observation_space=spaces.Box(low=lower_limits,high=upper_limits)
        self.action_space=spaces.Box(low=lower_limits,high=upper_limits)
        particles_upper_limits=[1 for _ in range(3)]+[100 for _ in range(3)] #distance and velocity vectors
        particles_lower_limits=[-1 for _ in range(3)]+[-100 for _ in range(3)]
        self.observation_space=spaces.Box(low=np.concatenate([particles_lower_limits for _ in range(n_particles)]),
            high=np.concatenate([particles_upper_limits for _ in range(n_particles)])
        )
        self.runtime_error_count=0

    def calculate_reward(self,observation:list):
        reward=0.0
        counts=0
        for [x,y,z,v_x,v_y,v_z] in observation:
            if np.linalg.norm([x,y])< self.nozzle_radius and z>=1:
                reward+=v_z #for each particle, that is in the nozzle, we want as much z momentumas possible
                counts+=1
        print("found rewards")
        return reward,counts
    
    def step(self,action):

        coil_list=[]
        regularization=0
        for c in range(self.n_coils):
            coil_parameters=action[c:c+self.parameters_per_coil]
            curve = CurveXYZFourier(1000, self.max_fourier_n)  # 100 = Number of quadrature points, 1 = max Fourier mode number
            fourier_amplitudes=coil_parameters[:-2]
            regularization+=self.regularization_lambda*sum([f**2 for f in fourier_amplitudes])
            z=coil_parameters[-2]
            new_x=np.concatenate((fourier_amplitudes, [z],[0. for _ in range(2*self.max_fourier_n)]) )
            curve.x =new_x  # Set Fourier amplitudes
            coil = Coil(curve, Current(coil_parameters[-1]))  # 10 kAmpere-turns
            coil_list.append(coil)
        #print("made coils")
        field=BiotSavart(coil_list)
        #print("calculated field")
        #field=InterpolatedField(field,4,[0,1,100],[0,2*np.pi,100],[0,1,100])
        #print("interpolated field")
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        Ekin = 10*ONE_EV
        #print("time to trace particles")
        try:
            res_tys,res_phi_hits=trace_particles(field,self.start_positions,self.start_velocities,mass=m,charge=q,Ekin=Ekin,mode="full",forget_exact_path=True,
                                                stopping_criteria=[MaxZStoppingCriterion(1),MinZStoppingCriterion(0), MaxRStoppingCriterion(self.radius)])
            
            #print("successfully traced particles :)))")
            observation=[rt[-1][1:] for rt in res_tys]
            reward,counts=self.calculate_reward(observation)
        except RuntimeError:
            reward=0
            self.runtime_error_count+=1
            observation=[[0 for _ in range(6)] for ___ in range(self.n_particles)]

        reward-=regularization        

        terminated=False
        if counts==self.n_particles:
            terminated=True #maybe NOT do this?

        truncated=False
        info={}
        observation=np.concatenate(observation)
        
        return observation, reward, terminated, truncated, info

    def reset(self,seed):
        return np.random.normal(0.5,size=(self.n_particles,7)),{}


if __name__=="__main__":
    #target_values=[[0.5,0.5,float(z)/10, 0,0,1] for z in range(10)] 
    print("making env???")
    env=MagneticOptimizationEnv(
        [[0,0,.1]],[1],2,1,0.25,1,0.001
    )
    # Train PPO agent
    print("made env")
    model = PPO("MlpPolicy", env, verbose=1)
    print("made model")
    start=time.time()
    model.learn(total_timesteps=2)
    print("errors",env.runtime_error_count)
    print(f"elapsed {time.time()-start} seconds")