import gym
from gym.wrappers import Monitor
import numpy as np
from stable_baselines3 import PPO
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import wandb


hyperparameter_defaults = {
    "n_envs": 16,
    "total_timesteps": int(1e6),
    "n_steps": 1024,
    "batch_size": 64,
    "gae_lambda": 0.98,
    "gamma": 0.999,
    "n_epochs": 4,
    "ent_coef": 0.01
}

wandb.init(project="ares-ea-rl",
           entity="msk-ipc",
           config=hyperparameter_defaults,
           sync_tensorboard=True,
           monitor_gym=True)

env = make_vec_env("LunarLanderContinuous-v2", n_envs=wandb.config["n_envs"])

eval_env = gym.make("LunarLanderContinuous-v2")
eval_env = Monitor(eval_env, f"recordings/{wandb.run.name}")

model = PPO("MlpPolicy",
            env,
            n_steps=wandb.config["n_steps"],
            batch_size=wandb.config["batch_size"],
            gae_lambda=wandb.config["gae_lambda"],
            gamma=wandb.config["gamma"],
            n_epochs=wandb.config["n_epochs"],
            ent_coef=wandb.config["ent_coef"],
            tensorboard_log=f"log/{wandb.run.name}",
            verbose=2)

model.learn(total_timesteps=wandb.config["total_timesteps"],
            log_interval=10,
            eval_env=eval_env,
            eval_freq=1000)

model.save(f"model_lunar_{wandb.run.name}")
