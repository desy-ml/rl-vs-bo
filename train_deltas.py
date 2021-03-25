from accelerator_environments.wrappers import NormalizeAction, NormalizeObservation, NormalizeReward
import gym
from gym.logger import WARN
from gym.wrappers import Monitor
import numpy as np
from stable_baselines3 import PPO
import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import wandb
from wandb import wandb_agent


hyperparameter_defaults = {
    "n_envs": 16,
    "total_timesteps": int(1e6),
    "n_steps": 50,
    "learning_rate": 0.0003,
    "batch_size": 64,
    "gae_lambda": 0.98,
    "gamma": 0.999,
    "n_epochs": 4,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [400, 300],
    "max_delta_quadrupole": 0.1,
    "max_delta_corrector": 0.05
}

wandb.init(project="ares-ea-rl",
           entity="msk-ipc",
           config=hyperparameter_defaults,
           sync_tensorboard=True,
           monitor_gym=True)

def make_environment():
    env = gym.make("ARESEA-JOSS-v0")
    env.action_space = gym.spaces.Box(low=-np.array([wandb.config["max_delta_quadrupole"]] * 3 + [wandb.config["max_delta_corrector"]] * 2),
                                      high=np.array([wandb.config["max_delta_quadrupole"]] * 3 + [wandb.config["max_delta_corrector"]] * 2))
    env = NormalizeAction(env)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    return env

vectorized = DummyVecEnv([make_environment for _ in range(wandb.config["n_envs"])])

eval_env = gym.make("ARESEA-JOSS-v0")
eval_env = NormalizeAction(eval_env)
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeReward(eval_env)
eval_env = Monitor(eval_env, f"recordings/{wandb.run.name}")

model = PPO("MlpPolicy",
            vectorized,
            n_steps=wandb.config["n_steps"],
            batch_size=wandb.config["batch_size"],
            learning_rate=wandb.config["learning_rate"],
            gae_lambda=wandb.config["gae_lambda"],
            gamma=wandb.config["gamma"],
            n_epochs=wandb.config["n_epochs"],
            clip_range=wandb.config["clip_range"],
            ent_coef=wandb.config["ent_coef"],
            vf_coef=wandb.config["vf_coef"],
            max_grad_norm=wandb.config["max_grad_norm"],
            policy_kwargs={"net_arch": wandb.config["net_arch"]},
            tensorboard_log=f"log/{wandb.run.name}",
            verbose=2)

model.learn(total_timesteps=wandb.config["total_timesteps"],
            log_interval=10,
            eval_env=eval_env,
            eval_freq=1000)

model.save(f"model_ppo_{wandb.run.name}")
