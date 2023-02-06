'''
Simple attempt to learn map-specific Q function for frozen lake. This means that
the state space is the precise location in the map.
'''

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import pdb

# Hyper-parameters of simulation
MAP_SIZE = 10

# Create and set environment
env = gym.make("FrozenLake-v1", render_mode="human",desc=generate_random_map(size=MAP_SIZE))
observation, info = env.reset(seed=42)

# pdb.set_trace()

# Create tabular Q-map
# 3D map: q_map(x,y,a) = q value
q_map = np.random.rand(MAP_SIZE,MAP_SIZE,env.action_space.n)






# for _ in range(10):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)
#    print(reward)
#    if terminated or truncated:
#       observation, info = env.reset()
# env.close()