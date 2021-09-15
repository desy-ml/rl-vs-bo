from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb

from environments.onestep import ARESEAOneStep


hyperparameter_defaults = {
    "total_timesteps": 600000
}

wandb.init(
    project="ares-ea-rl-one-step-at-a-time",
    entity="msk-ipc",
    config=hyperparameter_defaults,
    sync_tensorboard=True,
    settings=wandb.Settings(start_method="fork")
)

env = make_vec_env(ARESEAOneStep, n_envs=4)

model = PPO(
    "MlpPolicy",
    env,
    tensorboard_log=f"log/{wandb.run.name}",
    verbose=2
)

model.learn(
    total_timesteps=wandb.config["total_timesteps"],
    log_interval=10,
    eval_env=ARESEAOneStep(),
    eval_freq=10000
)

model.save(f"models/{wandb.run.name}")
