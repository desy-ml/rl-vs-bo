import base64
import os
import pickle
import subprocess
from datetime import datetime, timedelta

import gym
import numpy as np
import wandb
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def remove_if_exists(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix="rl_model",
        save_env=False,
        env_name_prefix="vec_normalize",
        save_replay_buffer=False,
        replay_buffer_name_prefix="replay_buffer",
        delete_old_replay_buffers=True,
        verbose=0,
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
            path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

            # Save env (VecNormalize wrapper)
            if self.save_env:
                path = os.path.join(
                    self.save_path,
                    f"{self.env_name_prefix}_{self.num_timesteps}_steps.pkl",
                )
                self.training_env.save(path)
                if self.verbose > 1:
                    print(f"Saving environment to {path[:-4]}")

            # Save replay buffer
            if self.save_replay_buffer:
                path = os.path.join(
                    self.save_path,
                    f"{self.replay_buffer_name_prefix}_{self.num_timesteps}_steps",
                )
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
            os.system(
                f"sbatch --export=ALL,WANDB_RESUME=allow,WANDB_RUN_ID={wandb.run.id} td3.sh"
            )
            if self.verbose > 1:
                print(f"Scheduling new batch job to continue training")
            return False
        else:
            if self.verbose > 1:
                print(
                    f"Continue running with this SLURM job (passed={passed_time} / allowed={self.allowed_time} / dt={dt})"
                )
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
            unfiltered = np.full(
                self.env.action_space.shape,
                self.replace,
                dtype=self.env.action_space.dtype,
            )

        unfiltered[self.filter_indicies] = action

        return unfiltered


class RecordEpisode(gym.Wrapper):
    """Wrapper for recording epsiode data such as observations, rewards, infos and actions."""

    def __init__(self, env):
        super().__init__(env)

        self.has_previously_run = False

    def reset(self):
        if self.has_previously_run:
            self.previous_observations = self.observations
            self.previous_rewards = self.rewards
            self.previous_infos = self.infos
            self.previous_actions = self.actions
            self.previous_t_start = self.t_start
            self.previous_t_end = datetime.now()
            self.previous_steps_taken = self.steps_taken

        observation = self.env.reset()

        self.observations = [observation]
        self.rewards = []
        self.infos = []
        self.actions = []
        self.t_start = datetime.now()
        self.t_end = None
        self.steps_taken = 0

        self.has_previously_run = True

        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.observations.append(observation)
        self.rewards.append(reward)
        self.infos.append(info)
        self.actions.append(action)
        self.steps_taken += 1

        return observation, reward, done, info


class PolishedDonkeyCompatibility(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    super().observation_space.low[4],
                    super().observation_space.low[5],
                    super().observation_space.low[7],
                    super().observation_space.low[6],
                    super().observation_space.low[8],
                    super().observation_space.low[9],
                    super().observation_space.low[11],
                    super().observation_space.low[10],
                    super().observation_space.low[12],
                    super().observation_space.low[0],
                    super().observation_space.low[2],
                    super().observation_space.low[1],
                    super().observation_space.low[3],
                ]
            ),
            high=np.array(
                [
                    super().observation_space.high[4],
                    super().observation_space.high[5],
                    super().observation_space.high[7],
                    super().observation_space.high[6],
                    super().observation_space.high[8],
                    super().observation_space.high[9],
                    super().observation_space.high[11],
                    super().observation_space.high[10],
                    super().observation_space.high[12],
                    super().observation_space.high[0],
                    super().observation_space.high[2],
                    super().observation_space.high[1],
                    super().observation_space.high[3],
                ]
            ),
        )

        self.action_space = spaces.Box(
            low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32) * 0.1,
            high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32) * 0.1,
        )

    def reset(self):
        return self.observation(super().reset())

    def step(self, action):
        observation, reward, done, info = super().step(self.action(action))
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return np.array(
            [
                observation[4],
                observation[5],
                observation[7],
                observation[6],
                observation[8],
                observation[9],
                observation[11],
                observation[10],
                observation[12],
                observation[0],
                observation[2],
                observation[1],
                observation[3],
            ]
        )

    def action(self, action):
        return np.array(
            [
                action[0],
                action[1],
                action[3],
                action[2],
                action[4],
            ]
        )


def send_to_elog(author, title, severity, text, elog, image=None):
    """Send information to a supplied electronic logbook."""

    # The DOOCS elog expects an XML string in a particular format. This string
    # is beeing generated in the following as an initial list of strings.
    succeded = True  # indicator for a completely successful job
    # list beginning
    elogXMLStringList = ['<?xml version="1.0" encoding="ISO-8859-1"?>', "<entry>"]
    # author information
    elogXMLStringList.append("<author>")
    elogXMLStringList.append(author)
    elogXMLStringList.append("</author>")
    # title information
    elogXMLStringList.append("<title>")
    elogXMLStringList.append(title)
    elogXMLStringList.append("</title>")
    # severity information
    elogXMLStringList.append("<severity>")
    elogXMLStringList.append(severity)
    elogXMLStringList.append("</severity>")
    # text information
    elogXMLStringList.append("<text>")
    elogXMLStringList.append(text)
    elogXMLStringList.append("</text>")
    # image information
    if image:
        try:
            encodedImage = base64.b64encode(image)
            elogXMLStringList.append("<image>")
            elogXMLStringList.append(encodedImage.decode())
            elogXMLStringList.append("</image>")
        except:  # make elog entry anyway, but return error (succeded = False)
            succeded = False
    # list end
    elogXMLStringList.append("</entry>")
    # join list to the final string
    elogXMLString = "\n".join(elogXMLStringList)
    # open printer process
    try:
        lpr = subprocess.Popen(
            ["/usr/bin/lp", "-o", "raw", "-d", elog],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        # send printer job
        lpr.communicate(elogXMLString.encode("utf-8"))
    except Exception as e:
        print(e)
        succeded = False
    return succeded


class NotVecNormalize(gym.Wrapper):
    """
    Normal Gym wrapper that replicates the functionality of Stable Baselines3's VecNormalize wrapper
    for non VecEnvs (i.e. `gym.Env`) in production.
    """

    def __init__(self, env, path):
        super().__init__(env)

        with open(path, "rb") as file_handler:
            self.vec_normalize = pickle.load(file_handler)

    def reset(self):
        observation = self.env.reset()
        return self.vec_normalize.normalize_obs(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.vec_normalize.normalize_obs(observation)
        reward = self.vec_normalize.normalize_reward(reward)
        return observation, reward, done, info
