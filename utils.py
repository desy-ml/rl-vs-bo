from datetime import datetime, timedelta
import os

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb


def remove_if_exists(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


class CheckpointCallback(BaseCallback):

    def __init__(self,
        save_freq,
        save_path,
        name_prefix = "rl_model",
        save_env = False,
        env_name_prefix = "vec_normalize",
        save_replay_buffer = False,
        replay_buffer_name_prefix = "replay_buffer",
        delete_old_replay_buffers = True,
        verbose = 0
    ):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_env = save_env
        self.env_name_prefix = env_name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.replay_buffer_name_prefix = replay_buffer_name_prefix
        self.delete_old_replay_buffers = delete_old_replay_buffers

    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
            
            # Save env (VecNormalize wrapper)
            if self.save_env:
                path = os.path.join(self.save_path, f"{self.env_name_prefix}_{self.num_timesteps}_steps.pkl")
                self.training_env.save(path)
                if self.verbose > 1:
                    print(f"Saving environment to {path[:-4]}")

            # Save replay buffer
            if self.save_replay_buffer:
                path = os.path.join(self.save_path, f"{self.replay_buffer_name_prefix}_{self.num_timesteps}_steps")
                self.model.save_replay_buffer(path)
                if self.verbose > 1:
                    print(f"Saving replay buffer to {path}")
                
                if self.delete_old_replay_buffers and hasattr(self, "last_saved_path"):
                    remove_if_exists(self.last_saved_path + ".pkl")
                    if self.verbose > 1:
                        print(f"Removing old replay buffer at {self.last_saved_path}")
                
                self.last_saved_path = path

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


class FilterAction(gym.ActionWrapper):

    def __init__(self, env, filter_indicies, replace="random"):
        super().__init__(env)

        self.filter_indicies = filter_indicies
        self.replace = replace

        self.action_space = spaces.Box(
            low=env.action_space.low[filter_indicies],
            high=env.action_space.high[filter_indicies],
            shape=env.action_space.low[filter_indicies].shape,
            dtype=env.action_space.dtype,
        )
    
    def action(self, action):
        if self.replace == "random":
            unfiltered = self.env.action_space.sample()
        else:
            unfiltered = np.full(self.env.action_space.shape, self.replace, dtype=self.env.action_space.dtype)
        
        unfiltered[self.filter_indicies] = action

        return unfiltered


class RecordData(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observations = None
        self.rewards = None
        self.infos = None
        self.actions = None

    def reset(self):
        self.previous_observations = self.observations
        self.previous_rewards = self.rewards
        self.previous_infos = self.infos
        self.previous_actions = self.actions

        observation = self.env.reset()

        self.observations = [observation]
        self.rewards = []
        self.infos = []
        self.actions = []

        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.observations.append(observation)
        self.rewards.append(reward)
        self.infos.append(info)
        self.actions.append(action)

        return observation, reward, done, info