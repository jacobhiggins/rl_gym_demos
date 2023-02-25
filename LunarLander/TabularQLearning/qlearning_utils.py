import pdb
import numpy as np
import gymnasium as gym
import re

def get_epsilon(i,epsilon_min,epsilon_max,training_duration):
    return float(i-training_duration+1)*(epsilon_min-epsilon_max)/float(training_duration-1) + epsilon_min

def cobs2dobs(cobs,observation_space,divs):
    '''
    Converts continuous observation to discrete observation
    cobs: continuous observation
    observation_space: provided by open ai gym env
    divs: number of divisions per obs dimension
    '''
    high_obs,low_obs = observation_space.high,observation_space.low
    delta = high_obs-low_obs
    dobs = np.multiply(np.divide(cobs-low_obs,delta),divs).astype(int)
    dobs = np.maximum(np.minimum(dobs,divs),np.zeros((divs.shape[0]))).astype(int)
    # pdb.set_trace()
    return dobs

def get_optimal_action(qmap,observation):
    return np.argmax(qmap[:,
                        observation[0],
                        observation[1],
                        observation[2],
                        observation[3],
                        observation[4],
                        observation[5],
                        observation[6],
                        observation[7]])

def update_qmap(qmap,observation_prev,observation_new,action,reward,GAMMA):
    qmap[action,
        observation_prev[0],
        observation_prev[1],
        observation_prev[2],
        observation_prev[3],
        observation_prev[4],
        observation_prev[5],
        observation_prev[6],
        observation_prev[7]] = reward + GAMMA*np.max(qmap[:,observation_new[0],
                                                            observation_new[1],
                                                            observation_new[2],
                                                            observation_new[3],
                                                            observation_new[4],
                                                            observation_new[5],
                                                            observation_new[6],
                                                            observation_new[7]])
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