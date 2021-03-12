import accelerator_environments
import gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common import callbacks
from stable_baselines3.common.noise import NormalActionNoise
import wandb


wandb.init(project="ares-ea-rl", entity="msk-ipc", sync_tensorboard=True, monitor_gym=True)

env = gym.make("ARESEA-JOSS-v0")
env = accelerator_environments.wrappers.NormalizeAction(env)
env = accelerator_environments.wrappers.NormalizeObservation(env)
env = accelerator_environments.wrappers.NormalizeReward(env)

env = gym.wrappers.Monitor(env, f"recordings/{wandb.run.name}")

n_actions = env.action_space.shape[-1]
noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=np.zeros(n_actions))
model = TD3("MlpPolicy",
            env,
            action_noise=noise,
            buffer_size=20000,
            tensorboard_log=f"log/{wandb.run.name}",
            verbose=2)

model.learn(total_timesteps=10000)

model.save("model_wandb")
