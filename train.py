import accelerator_environments
from accelerator_environments.wrappers import NormalizeAction, NormalizeObservation, NormalizeReward
import gym
from gym.wrappers import Monitor
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import wandb


hyperparameter_defaults = {
    "total_timesteps": int(3e5),
    "buffer_size": 200000,
    "learning_rate": 1e-3,
    "learning_starts": 10000,
    # "batch_size": 200,
    # "tau": 0.005,
    "gamma": 0.98,
    "gradient_steps": -1,
    # "policy_delay": 4,
    # "target_policy_noise": 0.05,
    # "target_noise_clip": 2.0,
    "action_noise_scale": 0.1,
    "net_arch": [400, 300]
}

wandb.init(project="ares-ea-rl",
           entity="msk-ipc",
           config=hyperparameter_defaults,
           sync_tensorboard=True,
           monitor_gym=True)

env = gym.make("ARESEA-JOSS-v0")
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = NormalizeReward(env)

eval_env = gym.make("ARESEA-JOSS-v0")
eval_env = NormalizeAction(eval_env)
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeReward(eval_env)
eval_env = Monitor(eval_env, f"recordings/{wandb.run.name}")

n_actions = env.action_space.shape[-1]
noise = NormalActionNoise(mean=np.zeros(n_actions),
                          sigma=np.full(n_actions, wandb.config["action_noise_scale"]))

model = TD3("MlpPolicy",
            env,
            buffer_size=wandb.config["buffer_size"],
            action_noise=noise,
            learning_rate=wandb.config["learning_rate"],
            learning_starts=wandb.config["learning_starts"],
            # batch_size=wandb.config["batch_size"],
            # tau=wandb.config["tau"],
            gamma=wandb.config["gamma"],
            gradient_steps=wandb.config["gradient_steps"],
            # policy_delay=wandb.config["policy_delay"],
            # target_policy_noise=wandb.config["target_policy_noise"],
            # target_noise_clip=wandb.config["target_noise_clip"],
            policy_kwargs={"net_arch": wandb.config["net_arch"]},
            tensorboard_log=f"log/{wandb.run.name}",
            verbose=2)

model.learn(total_timesteps=wandb.config["total_timesteps"],
            log_interval=10,
            eval_env=eval_env,
            eval_freq=1000)

model.save(f"model_zoo_parameters_{wandb.run.name}")
