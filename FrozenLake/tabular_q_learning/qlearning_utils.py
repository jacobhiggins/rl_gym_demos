import pdb
import numpy as np
import gymnasium as gym
import re

def get_hole_states(frozen_map):
    size = len(frozen_map)
    holes = []
    for (i,row) in enumerate(frozen_map):
        holes_row = [(m.start()+i*size) for m in re.finditer("H",row)]
        [holes.append(hole_idx) for hole_idx in holes_row]
    return holes

def get_optimal_action(qmap,observation):
    return np.argmax(qmap[observation,:])

def update_qmap(qmap,observation_prev,observation_new,action,reward,GAMMA):
    qmap[observation_prev,action] = reward + GAMMA*np.max(qmap[observation_new,:])
    return qmap

class BasicWrapper(gym.Wrapper):
    def __init__(self,env,holes):
        super().__init__(env)
        self.holes = np.array(holes)
    def step(self,action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if np.any(next_state==self.holes):
            reward = -10
        return next_state, reward, terminated, truncated, info