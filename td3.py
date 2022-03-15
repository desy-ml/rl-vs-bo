from datetime import datetime, timedelta
import glob
import os

from gym.wrappers import RescaleAction, TimeLimit
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb

from environments import ARESEASequential, ResetActuators


CHUNK_LENGTH = 3000
NODE_TIMEOUT = timedelta(hours=24)
SAFETY = timedelta(hours=1)

HYPERPARAMETER_DEFAULTS = {
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
    "net_arch": [64,32]
}


def make_env():
    env = ARESEASequential(
        backend="simulation",
        backendargs={"measure_beam": "direct"}
    )
    env = ResetActuators(env)
    env = TimeLimit(env, max_episode_steps=50)
    env = RescaleAction(env, -1, 1)
    env = Monitor(env)
    return env


def setup_new_training():
    wandb.config = HYPERPARAMETER_DEFAULTS

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=wandb.config["gamma"])

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
        verbose=1
    )

    return env, model


def find_resume_steps(name):
    models = glob.glob(f"log/{name}/*_model.zip")
    resume_steps = max(int(model.split("/")[-1][:-10]) for model in models)
    return resume_steps


def load_training(name):
    resume_steps = find_resume_steps(name)
    resume_path = f"log/{name}/{resume_steps}_"

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(resume_path + "vec_normalize.pkl", env)

    model = TD3.load(resume_path + "model.zip", env=env)
    model.load_replay_buffer(resume_path + "replay_buffer")

    return env, model


def remove_if_exists(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def main():
    wandb.init(
        project="ares-ea-rl-test",
        entity="msk-ipc",
        sync_tensorboard=True,
        settings=wandb.Settings(start_method="fork")
    )

    env, model = load_training(wandb.run.name) if wandb.run.resumed else setup_new_training()    

    t_start = datetime.now()
    while True:
        t_last = datetime.now()
        last_total_timesteps = model._total_timesteps

        model.learn(
            total_timesteps=CHUNK_LENGTH,
            reset_num_timesteps=False,
            log_interval=1,
            tb_log_name="TD3"
        )

        model.save(f"log/{wandb.run.name}/{model._total_timesteps}_model")
        model.save_replay_buffer(f"log/{wandb.run.name}/{model._total_timesteps}_replay_buffer")
        env.save(f"log/{wandb.run.name}/{model._total_timesteps}_vec_normalize.pkl")

        remove_if_exists(f"log/{wandb.run.name}/{last_total_timesteps}_replay_buffer.pkl")

        # Is enough time left for another iteration?
        chunk_time = datetime.now() - t_last
        allowed_run_time = NODE_TIMEOUT - SAFETY
        passed_run_time = datetime.now() - t_start
        if passed_run_time + chunk_time > allowed_run_time:
            os.system(f"sbatch --export=ALL,WANDB_RESUME=allow,WANDB_RUN_ID={wandb.run.id} td3.sh")
            break


if __name__ == "__main__":
    main()

