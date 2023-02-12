'''
Simple attempt to learn map-specific Q function for frozen lake. This means that
the state space is the precise location in the map.
'''

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import pdb
import qlearning_utils
import pickle
import tqdm

# Hyper-parameters of simulation
MAP_SIZE = 15
TRAINING_DURATION = 500
GAMMA = 0.5
EPSILON=0.99
MAP_SEED = 0
SAVE_POLICY = True

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
training_iter = 0

for _ in tqdm.tqdm(range(TRAINING_DURATION)):
    terminated = False
    while not terminated:
        # Get (currently) optimal action EPSILON% of the time, else use random action
        action = qlearning_utils.get_optimal_action(qmap,observation_prev) if np.random.rand()>EPSILON else env.action_space.sample()
        # Perform action
        observation_new, reward, terminated, truncated, info = env.step(action)
        # Update qmap, return value for observation_prev and action
        qmap = qlearning_utils.update_qmap(qmap,observation_prev,observation_new,action,reward,GAMMA)
        # Update prev observation
        observation_prev = observation_new
    # If terminated, reset the environment and update epsilon
    observation_prev, info = env.reset()
    EPSILON*=EPSILON

# while training_iter < TRAINING_DURATION:
#     # Get (currently) optimal action EPSILON% of the time, else use random action
#     action = qlearning_utils.get_optimal_action(qmap,observation_prev) if np.random.rand()>EPSILON else env.action_space.sample()

#     # Perform action
#     observation_new, reward, terminated, truncated, info = env.step(action)
#     # Update qmap, return value for observation_prev and action
#     qmap = qlearning_utils.update_qmap(qmap,observation_prev,observation_new,action,reward,GAMMA)

#     if terminated or truncated:
#         # print("Truncated!") if truncated else print("Terminated!")
#         observation_prev, info = env.reset()
#         training_iter+=1
#         EPSILON*=EPSILON
#     else:
#         observation_prev = observation_new

# Testing the policy
print("TESTING OPTIMAL POLICY")
terminated = False
observation, info = env.reset(seed=42)
while not terminated:
    action = qlearning_utils.get_optimal_action(qmap,observation)
    observation, reward, terminated, truncated, info = env.step(action)

# Save Policy
if SAVE_POLICY:
    policy_file_name = "./optimal_policy.pkl"
    pickle.dump((MAP_SIZE,MAP_SEED,qmap),open(policy_file_name,"wb"))