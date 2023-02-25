'''
The approach here is to use tabular Q learning.
State and action space is discretized 
'''
import gymnasium as gym
import pdb
import qlearning_utils
import numpy as np
import tqdm

'''
Observation space:
 - (x,y) position of lundar lander
 - (x,y) velocities of lunar lander
 - angle
 - angular velocity
 - 2 booleans, legs connected with ground
'''

# Training params
training_duration = 1000
epsilon_max = 1.00
epsilon_min = 0.05
gamma = 0.9
map_seed = 0
save_policy = True

divs = [
    15, # x position
    10, # y position
    5, # x velocity
    5, # y velocity
    10, # angle
    10, # angular velocity
    1, # left/right leg contact
    1  # left/right leg contact
]
divs = np.array(divs)

# Create env
env = gym.make("LunarLander-v2", render_mode="human")
cobs, info = env.reset(seed=42)

# Create tabular q function
qmap = np.zeros(tuple(np.append(np.array(env.action_space.n),divs+1)))

for i in tqdm.tqdm(range(training_duration)):
    terminated = False
    epsilon = qlearning_utils.get_epsilon(i,epsilon_min,epsilon_max,training_duration)
    # print(epsilon)
    while not terminated:
        dobs = qlearning_utils.cobs2dobs(cobs,env.observation_space,divs)
        # Get (currently) optimal action EPSILON% of the time, else use random action
        action = qlearning_utils.get_optimal_action(qmap,dobs) if np.random.rand()>epsilon else env.action_space.sample()
        # Perform action
        cobs_new, reward, terminated, truncated, info = env.step(action)
        dobs_new = qlearning_utils.cobs2dobs(cobs_new,env.observation_space,divs)
        # Update qmap, return value for observation_prev and action
        qmap = qlearning_utils.update_qmap(qmap,dobs,dobs_new,action,reward,gamma)
        # Update prev observation
        dobs = dobs_new
    # If terminated, reset the environment and update epsilon
    cobs, info = env.reset()
    dobs = qlearning_utils.cobs2dobs(cobs,env.observation_space,divs)

# dobs = qlearning_utils.cobs2dobs(cobs,env.observation_space,divs)
env.close()

# pdb.set_trace()