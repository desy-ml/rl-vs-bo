import accelerator_environments
import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


env = gym.make("ARESExperimentalArea-Ocelot-v0")

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG("CnnPolicy", env, buffer_size=20000, verbose=2)
model.learn(total_timesteps=10000)

model.save("model_maxwell")
