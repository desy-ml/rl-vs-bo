import argparse
import glob

from gym.wrappers import FrameStack, RescaleAction, TimeLimit
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from environments import ARESEAMisalignments
from utils import CheckpointCallback


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--runid", type=str, default=None, help="W&B run id to resume")

    args = parser.parse_args()
    return args.runid


def make_env():
    env = ARESEAMisalignments()
    env = FrameStack(env, 16)
    env = TimeLimit(env, max_episode_steps=50)
    env = RescaleAction(env, -1, 1)
    env = Monitor(env)
    return env


def setup_new_training():
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = SAC(
        "MlpPolicy",
        env,
        tensorboard_log=f"log2/{wandb.run.name}",
        verbose=1,
        device="cpu"
    )

    return model


def find_resume_steps(log_path):
    paths = glob.glob(f"{log_path}/rl_model_*_steps.zip")
    resume_steps = max(int(path.split("/")[-1].split("_")[-2]) for path in paths)
    return resume_steps


def load_training(log_path):
    resume_steps = find_resume_steps(log_path)

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(f"{log_path}/vec_normalize_{resume_steps}_steps.pkl", env)

    model = SAC.load(f"{log_path}/rl_model_{resume_steps}_steps.zip", env=env, device="cpu")
    model.load_replay_buffer(f"{log_path}/replay_buffer_{resume_steps}_steps.pkl")

    return model


def main():
    run_id = parse_arguments()
    if run_id is None:
        run_id = wandb.util.generate_id()

    wandb.init(
        project="ares-ea-rl-test",
        entity="msk-ipc",
        sync_tensorboard=True,
        id=run_id,
        resume="allow"
    )
    
    log_path = f"log2/{wandb.run.name}"

    model = load_training(log_path) if wandb.run.resumed else setup_new_training()

    callbacks = [
        CheckpointCallback(3000, log_path, save_env=True, save_replay_buffer=True),
        WandbCallback(verbose=1)
    ]

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    model.learn(
        total_timesteps=int(1e10),
        reset_num_timesteps=False,
        callback=callbacks,
        eval_env=eval_env,
        eval_freq=3000,
        tb_log_name="SAC"
    )


if __name__ == "__main__":
    main()
