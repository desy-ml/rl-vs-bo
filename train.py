import accelerator_environments
import gym
from gym.wrappers import Monitor
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import wandb


hyperparameter_defaults = {
    "total_timesteps": 600000,
    "buffer_size": 600000,
    "learning_rate": 1e-3,
    "learning_starts": 2000,
    "gamma": 0.8,
    "action_noise_scale": 0.1,
    "net_arch": [64, 32]
}

wandb.init(project="ares-ea-rl-a-new-hope",
           entity="msk-ipc",
           config=hyperparameter_defaults,
           sync_tensorboard=True,
           monitor_gym=True,
           settings=wandb.Settings(start_method="fork"))

env = gym.make("ARESEA-JOSS-v0")

eval_env = gym.make("ARESEA-JOSS-v0")
eval_env = Monitor(eval_env, f"recordings/{wandb.run.name}", video_callable=lambda i: (i % 5) == 0)

n_actions = env.action_space.shape[-1]
noise = NormalActionNoise(mean=np.zeros(n_actions),
                          sigma=np.full(n_actions, wandb.config["action_noise_scale"]))

model = TD3("MlpPolicy",
            env,
            buffer_size=wandb.config["buffer_size"],
            action_noise=noise,
            learning_rate=wandb.config["learning_rate"],
            learning_starts=wandb.config["learning_starts"],
            gamma=wandb.config["gamma"],
            policy_kwargs={"net_arch": wandb.config["net_arch"]},
            tensorboard_log=f"log/{wandb.run.name}",
            verbose=2)

model.learn(total_timesteps=wandb.config["total_timesteps"],
            log_interval=10,
            eval_env=eval_env,
            eval_freq=10000)

model.save(f"models/model-{wandb.run.name}")
