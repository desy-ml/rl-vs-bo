import multiprocessing

from gym.wrappers import RescaleAction, TimeLimit
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import wandb

from environments import ARESEASequential, ResetActuators


def main():
    hyperparameter_defaults = {
        "total_timesteps": 100000000,
        "learning_rate": 1e-3,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.55,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "use_sde": False,
        "sde_sample_freq": - 1,
        "net_arch": [64, 32]
    }

    wandb.init(
        project="ares-ea-rl-a-new-hope",
        entity="msk-ipc",
        config=hyperparameter_defaults,
        sync_tensorboard=True,
        settings=wandb.Settings(start_method="fork")
    )

    def make_env():
        env = ARESEASequential(
            backend="simulation",
            backendargs={"measure_beam": "direct"}
        )
        env = ResetActuators(env)
        env = TimeLimit(env, max_episode_steps=50)
        env = RescaleAction(env, -1, 1)
        return env

    env = SubprocVecEnv([make_env]*multiprocessing.cpu_count())
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=wandb.config["gamma"])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=wandb.config["learning_rate"],
        n_steps=wandb.config["n_steps"],
        batch_size=wandb.config["batch_size"],
        n_epochs=wandb.config["n_epochs"],
        gamma=wandb.config["gamma"],
        gae_lambda=wandb.config["gae_lambda"],
        clip_range=wandb.config["clip_range"],
        clip_range_vf=wandb.config["clip_range_vf"],
        ent_coef=wandb.config["ent_coef"],
        vf_coef=wandb.config["vf_coef"],
        use_sde=wandb.config["use_sde"],
        sde_sample_freq=wandb.config["sde_sample_freq"],
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


if __name__ == "__main__":
    main()
