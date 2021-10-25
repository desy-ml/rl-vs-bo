from gym.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb

from environments.onestep import ARESEAOneStep


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

def make_env():
    env = ARESEAOneStep(
        backend="simulation",
        random_incoming=True,
        random_initial=True,
        beam_parameter_method="direct"
    )
    env = RescaleAction(env, -1, 1)
    return env

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs={"net_arch": wandb.config["net_arch"]},
    tensorboard_log=f"log/{wandb.run.name}",
    verbose=2
)

model.learn(
    total_timesteps=wandb.config["total_timesteps"],
    log_interval=10
)

log_dir = f"models/{wandb.run.name}"
model.save(f"{log_dir}/model")
env.save(f"{log_dir}/vec_normalize.pkl")
