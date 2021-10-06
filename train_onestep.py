from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb

from environments.onestep_ppo import ARESEAOneStep


hyperparameter_defaults = {
    "total_timesteps": 600000,
    "net_arch": [64, 32]
}

wandb.init(
    project="ares-ea-rl-one-step-at-a-time",
    entity="msk-ipc",
    config=hyperparameter_defaults,
    sync_tensorboard=True,
    settings=wandb.Settings(start_method="fork")
)

# env = make_vec_env(ARESEAOneStep, n_envs=4)
env = ARESEAOneStep(
    backend="simulation",
    random_incoming=True,
    random_initial=True,
    beam_parameter_method="direct"
)

eval_env = ARESEAOneStep(
    backend="simulation",
    random_incoming=True,
    random_initial=True,
    beam_parameter_method="direct"
)

model = PPO(
    "MlpPolicy",
    env,
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
