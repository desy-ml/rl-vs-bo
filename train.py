import accelerator_environments
import gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common import callbacks
from stable_baselines3.common.noise import NormalActionNoise
import wandb


# feasible-sweep-1935
hyperparameter_defaults = {
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 200,
    "tau": 0.005,
    "gamma": 0.8,
    "policy_delay": 4,
    "target_policy_noise": 0.05,
    "target_noise_clip": 2.0,
    "action_noise_scale": 0.1,
    "net_arch": [32, 32]
}

wandb.init(project="ares-ea-rl",
           entity="msk-ipc",
           config=hyperparameter_defaults,
           sync_tensorboard=True,
           monitor_gym=True)

env = gym.make("ARESEA-JOSS-v0")
env = accelerator_environments.wrappers.NormalizeAction(env)
env = accelerator_environments.wrappers.NormalizeObservation(env)
env = accelerator_environments.wrappers.NormalizeReward(env)

env = gym.wrappers.Monitor(env, f"recordings/{wandb.run.name}")

n_actions = env.action_space.shape[-1]
noise = NormalActionNoise(mean=np.zeros(n_actions),
                          sigma=np.full(n_actions, wandb.config["action_noise_scale"]))

model = TD3("MlpPolicy",
            env,
            action_noise=noise,
            learning_rate=wandb.config["learning_rate"],
            learning_starts=wandb.config["learning_starts"],
            batch_size=wandb.config["batch_size"],
            tau=wandb.config["tau"],
            gamma=wandb.config["gamma"],
            policy_delay=wandb.config["policy_delay"],
            target_policy_noise=wandb.config["target_policy_noise"],
            target_noise_clip=wandb.config["target_noise_clip"],
            policy_kwargs={"net_arch": wandb.config["net_arch"]},
            tensorboard_log=f"log/{wandb.run.name}",
            verbose=2)

model.learn(total_timesteps=15000, log_interval=10)

model.save(f"model_sweep2nd_{wandb.run.name}")
