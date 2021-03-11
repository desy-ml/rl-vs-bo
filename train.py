import accelerator_environments
import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


env = gym.make("ARESExperimentalArea-JOSS-v0")
env = accelerator_environments.wrappers.NormalizeAction(env)
env = accelerator_environments.wrappers.NormalizeObservation(env)

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG("MlpPolicy", env, action_noise=action_noise, buffer_size=20000, verbose=2)
model.learn(total_timesteps=800, log_interval=1)

model.save("model_pang")
