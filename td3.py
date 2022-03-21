from copy import deepcopy
from datetime import datetime, timedelta
import glob
import os

from gym.wrappers import RescaleAction, TimeLimit
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb

from environments import ARESEASequential, ResetActuators


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


def remove_if_exists(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


class ReplayBufferCheckpointCallback(BaseCallback):

    def __init__(self, save_freq, save_path, delete_old=True, name_prefix="replay_buffer", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.delete_old = delete_old
        self.name_prefix = name_prefix
        self.last_saved_path = None
    
    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save_replay_buffer(path)
            if self.verbose > 1:
                print(f"Saving replay buffer to {path}")
            
            if self.delete_old and self.last_saved_path is not None:
                remove_if_exists(self.last_saved_path + ".pkl")
                if self.verbose > 1:
                    print(f"Removing old replay buffer at {self.last_saved_path}")
            
            self.last_saved_path = path

        return True


class EnvironmentCheckpointCallback(BaseCallback):

    def __init__(self, save_freq, save_path, name_prefix="environment", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_saved_path = None
    
    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.training_env.save(path)
            if self.verbose > 1:
                print(f"Saving environment to {path[:-4]}")

        return True


class SLURMRescheduleCallback(BaseCallback):

    def __init__(self, reserved_time, safety=timedelta(minutes=1), verbose=0):
        super().__init__(verbose)
        self.allowed_time = reserved_time - safety
        self.t_start = datetime.now()
        self.t_last = self.t_start
    
    def _on_step(self):
        t_now = datetime.now()
        passed_time = t_now - self.t_start
        dt = t_now - self.t_last
        self.t_last = t_now
        if passed_time + dt > self.allowed_time:
            os.system(f"sbatch --export=ALL,WANDB_RESUME=allow,WANDB_RUN_ID={wandb.run.id} td3.sh")
            if self.verbose > 1:
                print(f"Scheduling new batch job to continue training")
            return False
        else:
            if self.verbose > 1:
                print(f"Continue running with this SLURM job (passed={passed_time} / allowed={self.allowed_time} / dt={dt})")
            return True


def main():
    wandb.init(
        project="ares-ea-rl-test",
        entity="msk-ipc",
        sync_tensorboard=True,
        settings=wandb.Settings(start_method="thread")
    )
    
    log_path = f"log/{wandb.run.name}"

    model = load_training(log_path) if wandb.run.resumed else setup_new_training()

    callback = EveryNTimesteps(3000, callback=CallbackList([
        CheckpointCallback(1, log_path, verbose=2),
        ReplayBufferCheckpointCallback(1, log_path, verbose=2),
        EnvironmentCheckpointCallback(1, log_path, name_prefix="vec_normalize", verbose=2),
        # SLURMRescheduleCallback(timedelta(minutes=10), safety=timedelta(minutes=1), verbose=2)
    ]))

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, gamma=wandb.config["gamma"], training=False)

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
