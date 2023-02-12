import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import qlearning_utils
import pickle

# Import map params and optimal policy
policy_file_name = "./optimal_policy.pkl"
(MAP_SIZE,MAP_SEED,qmap) = pickle.load(open(policy_file_name,"rb"))

# Create env
frozen_map = generate_random_map(size=MAP_SIZE,seed=MAP_SEED)
env = gym.make("FrozenLake-v1", render_mode="human",desc=frozen_map, is_slippery=False)

# Perform optimal policy
terminated = False
observation, info = env.reset(seed=42)
while not terminated:
    action = qlearning_utils.get_optimal_action(qmap,observation)
    observation, reward, terminated, truncated, info = env.step(action)