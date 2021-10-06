from gym.wrappers import Monitor, TimeLimit, FlattenObservation
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import wandb

from environments.sequential import ARESEATransverseBeamSequential


hyperparameter_defaults = {
    "total_timesteps": 600000,
    "noise_type": "normal",
    "noise_std": 0.1,
    "learning_rate": 1e-3,
    "buffer_size": 600000,
    "learning_starts": 2000,
    "batch_size": 100,
    "tau": 0.005,
    "gamma": 0.55,
    "gradient_steps": -1,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "net_arch": [64, 32]
}

wandb.init(
    project="ares-ea-rl-a-new-hope",
    entity="msk-ipc",
    config=hyperparameter_defaults,
    sync_tensorboard=True,
    monitor_gym=True,
    settings=wandb.Settings(start_method="fork")
)

env = ARESEATransverseBeamSequential(
    backend="simulation",
    random_incoming=True,
    random_initial=True,
    beam_parameter_method="direct"
)
env = TimeLimit(env, max_episode_steps=50)
env = FlattenObservation(env)

eval_env = ARESEATransverseBeamSequential(
    backend="simulation",
    random_incoming=True,
    random_initial=True,
    beam_parameter_method="direct"
)
eval_env = TimeLimit(eval_env, max_episode_steps=50)
eval_env = FlattenObservation(eval_env)
eval_env = Monitor(eval_env, f"recordings/{wandb.run.name}", video_callable=lambda i: (i % 5) == 0)

n_actions = env.action_space.shape[-1]
if wandb.config["noise_type"] == "none":
    noise = None
elif wandb.config["noise_type"] == "normal":
    noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=np.full(n_actions, wandb.config["noise_std"])
    )

model = TD3(
    "MlpPolicy",
    env,
    action_noise=noise,
    learning_rate=wandb.config["learning_rate"],
    buffer_size=wandb.config["buffer_size"],
    learning_starts=wandb.config["learning_starts"],
    batch_size=wandb.config["batch_size"],
    tau=wandb.config["tau"],
    gamma=wandb.config["gamma"],
    gradient_steps=wandb.config["gradient_steps"],
    policy_delay=wandb.config["policy_delay"],
    target_policy_noise=wandb.config["target_policy_noise"],
    target_noise_clip=wandb.config["target_noise_clip"],
    policy_kwargs={"net_arch": wandb.config["net_arch"]},
    tensorboard_log=f"log/{wandb.run.name}",
    verbose=2
)

model.learn(
    total_timesteps=wandb.config["total_timesteps"],
    log_interval=10,
    eval_env=eval_env,
    eval_freq=10000
)

model.save(f"models/{wandb.run.name}")
