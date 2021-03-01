import accelerator_environments
import gym
from stable_baselines3 import DDPG


env = gym.make("ARESExperimentalArea-Ocelot-v0")

model = DDPG("CnnPolicy", env, buffer_size=20000, verbose=2)
model.learn(total_timesteps=10000)

model.save("model_maxwell")
