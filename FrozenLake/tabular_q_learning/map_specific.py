'''
Simple attempt to learn map-specific Q function for frozen lake. This means that
the state space is the precise location in the map.
'''

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import pdb
import qlearning_utils

# Hyper-parameters of simulation
MAP_SIZE = 6
TRAINING_DURATION = 500
GAMMA = 0.5
EPSILON=0.99
MAP_SEED = 0

# Generate random map, get holes in map
frozen_map = generate_random_map(size=MAP_SIZE,seed=MAP_SEED)
holes = qlearning_utils.get_hole_states(frozen_map)

# Create and set environment
env = gym.make("FrozenLake-v1", render_mode="human",desc=frozen_map, is_slippery=False)
env = qlearning_utils.BasicWrapper(env,holes)
observation_prev, info = env.reset(seed=42)

# Create tabular Q-map
# 3D map: q_map(observation,a) = q value
qmap = np.random.rand(MAP_SIZE*MAP_SIZE,env.action_space.n)

# Training the policy

for _ in range(TRAINING_DURATION):
    # Get (currently) optimal action 50% of the time, else use random action
    action = qlearning_utils.get_optimal_action(qmap,observation_prev) if np.random.rand()>EPSILON else env.action_space.sample()

    # Perform action
    observation_new, reward, terminated, truncated, info = env.step(action)
    print(reward)
    # Update qmap, return value for observation_prev and action
    qmap = qlearning_utils.update_qmap(qmap,observation_prev,observation_new,action,reward,GAMMA)

    if terminated or truncated:
        print("Truncated!") if truncated else print("Terminated!")
        observation_prev, info = env.reset()
    else:
        observation_prev = observation_new

    EPSILON*=EPSILON

# Testing the policy
print("TESTING OPTIMAL POLICY")
terminated = False
observation, info = env.reset(seed=42)
while not terminated:
    action = qlearning_utils.get_optimal_action(qmap,observation)
    observation, reward, terminated, truncated, info = env.step(action)