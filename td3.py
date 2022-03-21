import argparse
import glob

from gym.wrappers import RescaleAction, TimeLimit
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb

from environments import ARESEASequential, ResetActuators
from utils import CheckpointCallback


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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--runid", type=str, default=None, help="W&B run id to resume")

    args = parser.parse_args()
    return args.runid


def make_env():
    env = ARESEASequential(
        backend="simulation",
        backendargs={"measure_beam": "direct"}
    )
    env = ResetActuators(env)
    env = TimeLimit(env, max_episode_steps=50)
    env = RescaleAction(env, -1, 1)
    env = Monitor(env, info_keywords=("mae",))
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

    model = TD3.load(f"{log_path}/rl_model_{resume_steps}_steps.zip", env=env, device="cpu")
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
    
    log_path = f"log/{wandb.run.name}"

    model = load_training(log_path) if wandb.run.resumed else setup_new_training()

    callback = CheckpointCallback(3000, log_path, save_env=True, save_replay_buffer=True)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    model.learn(
        total_timesteps=int(1e10),
        reset_num_timesteps=False,
        callback=callback,
        eval_env=eval_env,
        eval_freq=3000,
        tb_log_name="TD3"
    )


if __name__ == "__main__":
    main()
