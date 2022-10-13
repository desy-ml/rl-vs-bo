import base64
import os
import pickle
import subprocess
from datetime import datetime, timedelta
from io import BytesIO

import gym
import matplotlib.pyplot as plt
import numpy as np
import wandb
from gym import spaces
from gym.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import is_wrapped, unwrap_wrapper
from tqdm import tqdm


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
    """
    Wrapper for recording epsiode data such as observations, rewards, infos and actions.
    Pass a `save_dir` other than `None` to save the recorded data to pickle files.
    """

    def __init__(self, env, save_dir=None, name_prefix="recorded_episode"):
        super().__init__(env)

        self.save_dir = save_dir
        if self.save_dir is not None:
            self.save_dir = os.path.abspath(save_dir)
            if os.path.isdir(self.save_dir):
                print(
                    f"Overwriting existing data recordings at {self.save_dir} folder. Specify a different `save_dir` for the `RecordEpisode` wrapper if this is not desired."
                )
            os.makedirs(self.save_dir, exist_ok=True)

        self.name_prefix = name_prefix

        self.n_episodes_recorded = 0

    def reset(self):
        self.t_end = datetime.now()

        if self.save_dir is not None and self.n_episodes_recorded > 0:
            self.save_to_file()

        if self.n_episodes_recorded > 0:
            self.previous_observations = self.observations
            self.previous_rewards = self.rewards
            self.previous_infos = self.infos
            self.previous_actions = self.actions
            self.previous_t_start = self.t_start
            self.previous_t_end = self.t_end
            self.previous_steps_taken = self.steps_taken

        self.n_episodes_recorded += 1

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

    def close(self):
        super().close()

        self.t_end = datetime.now()

        if self.save_dir is not None and self.n_episodes_recorded > 0:
            self.save_to_file()

    def save_to_file(self):
        """Save the data from the current episodes to a `.pkl` file."""
        filename = f"{self.name_prefix}_{self.n_episodes_recorded}.pkl"
        path = os.path.join(self.save_dir, filename)

        d = {
            "observations": self.observations,
            "rewards": self.rewards,
            "infos": self.infos,
            "actions": self.actions,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "steps_taken": self.steps_taken,
        }

        with open(path, "wb") as f:
            pickle.dump(d, f)


class ARESEAeLog(gym.Wrapper):
    """
    Wrapper to send a summary of optimsations in the ARES Experimental Area
    to the ARES eLog.
    """

    def __init__(self, env, model_name):
        super().__init__(env)

        self.model_name = model_name
        self.has_reset_before = False

    def reset(self):
        if self.has_reset_before:
            self.t_end = datetime.now()
            self.report_optimization_to_elog()
        else:
            self.has_reset_before = True

        observation = self.env.reset()
        self.beam_image_before = self.env.get_beam_image()

        self.observations = [observation]
        self.rewards = []
        self.infos = []
        self.actions = []
        self.t_start = datetime.now()
        self.t_end = None
        self.steps_taken = 0

        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.observations.append(observation)
        self.rewards.append(reward)
        self.infos.append(info)
        self.actions.append(action)
        self.steps_taken += 1

        return observation, reward, done, info

    def close(self):
        super().close()

        if self.has_reset_before:
            self.t_end = datetime.now()
            self.report_optimization_to_elog()

    def report_optimization_to_elog(self):
        """
        Send a summary report of the optimisation in the ARES EA environment to the ARES
        eLog.
        """
        msg = self.create_text_message()
        img = self.create_plot_jpg()
        title = "Beam Optimisation on AREABSCR1 using " + (
            "Bayesian Optimisation"
            if self.model_name == "Bayesian Optimisation"
            else "Reinforcement Learning"
        )

        print(f"{title = }")
        print(f"{msg = }")

        send_to_elog(
            elog="areslog",
            author="Autonomous ARES",
            title=title,
            severity="NONE",
            text=msg,
            image=img,
        )

    def create_text_message(self):
        """Create text message summarising the optimisation."""
        beam_before = self.infos[0][
            "beam_before_reset"
        ]  # TODO this may become an issue when magnet_init_values is None
        beam_after = self.observations[-1]["beam"]
        target_beam = self.observations[0]["target"]
        target_threshold = np.array(
            [
                self.env.target_mu_x_threshold,
                self.env.target_sigma_x_threshold,
                self.env.target_mu_y_threshold,
                self.env.target_sigma_y_threshold,
            ]
        )
        final_magnets = self.observations[-1]["magnets"]
        steps_taken = len(self.observations) - 1
        success = np.abs(beam_after - target_beam) < target_threshold

        return f"""{'Bayesian Optimisation' if self.model_name == 'Bayesian Optimisation' else 'Reinforcement Learning agent'} optimised beam on AREABSCR1

Agent: {self.model_name}
Start time: {self.t_start}
Time taken: {self.t_end - self.t_start}
No. of steps: {steps_taken}

Beam before:
    mu_x    = {beam_before[0] * 1e3: 5.4f} mm
    sigma_x = {beam_before[1] * 1e3: 5.4f} mm
    mu_y    = {beam_before[2] * 1e3: 5.4f} mm
    sigma_y = {beam_before[3] * 1e3: 5.4f} mm

Beam after:
    mu_x    = {beam_after[0] * 1e3: 5.4f} mm
    sigma_x = {beam_after[1] * 1e3: 5.4f} mm
    mu_y    = {beam_after[2] * 1e3: 5.4f} mm
    sigma_y = {beam_after[3] * 1e3: 5.4f} mm

Target beam:
    mu_x    = {target_beam[0] * 1e3: 5.4f} mm    (e = {target_threshold[0] * 1e3:5.4f} mm) {';)' if success[0] else ':/'}
    sigma_x = {target_beam[1] * 1e3: 5.4f} mm    (e = {target_threshold[1] * 1e3:5.4f} mm) {';)' if success[1] else ':/'}
    mu_y    = {target_beam[2] * 1e3: 5.4f} mm    (e = {target_threshold[2] * 1e3:5.4f} mm) {';)' if success[2] else ':/'}
    sigma_y = {target_beam[3] * 1e3: 5.4f} mm    (e = {target_threshold[3] * 1e3:5.4f} mm) {';)' if success[3] else ':/'}

Final magnet settings:
    AREAMQZM1 strength = {final_magnets[0]: 8.4f} 1/m^2
    AREAMQZM2 strength = {final_magnets[1]: 8.4f} 1/m^2
    AREAMCVM1 kick     = {final_magnets[2] * 1e3: 8.4f} mrad
    AREAMQZM3 strength = {final_magnets[3]: 8.4f} 1/m^2
    AREAMCHM1 kick     = {final_magnets[4] * 1e3: 8.4f} mrad
    """

    def create_plot_jpg(self):
        """Create plot overview of the optimisation and return it as jpg bytes."""
        fig, axs = plt.subplots(1, 5, figsize=(30, 4))
        plot_quadrupole_history(
            axs[0],
            self.observations,
            before_reset=self.infos[0]["magnets_before_reset"],
        )
        plot_steerer_history(
            axs[1],
            self.observations,
            before_reset=self.infos[0]["magnets_before_reset"],
        )
        plot_beam_history(
            axs[2], self.observations, before_reset=self.infos[0]["beam_before_reset"]
        )
        plot_beam_image(
            axs[3],
            self.infos[0][
                "screen_before_reset"
            ],  # TODO this may become an issue when magnet_init_values is None
            screen_resolution=self.infos[0]["screen_resolution"],
            pixel_size=self.infos[0]["pixel_size"],
            title="Beam at Reset (Background Removed)",
        )
        plot_beam_image(
            axs[4],
            self.infos[-1]["beam_image"],
            screen_resolution=self.infos[-1]["screen_resolution"],
            pixel_size=self.infos[-1]["pixel_size"],
            title="Beam After (Background Removed)",
        )
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, dpi=300, format="jpg")
        buf.seek(0)
        img = bytes(buf.read())

        return img


def plot_quadrupole_history(ax, observations, before_reset=None):
    areamqzm1 = [obs["magnets"][0] for obs in observations]
    areamqzm2 = [obs["magnets"][1] for obs in observations]
    areamqzm3 = [obs["magnets"][3] for obs in observations]

    if before_reset is not None:
        areamqzm1 = [before_reset[0]] + areamqzm1
        areamqzm2 = [before_reset[1]] + areamqzm2
        areamqzm3 = [before_reset[3]] + areamqzm3

    start = 0 if before_reset is None else -1
    steps = np.arange(start, len(observations))

    ax.set_title("Quadrupoles")
    ax.set_xlim([start, len(observations) + 1])
    ax.set_xlabel("Step")
    ax.set_ylabel("Strength (1/m^2)")
    ax.plot(steps, areamqzm1, label="AREAMQZM1")
    ax.plot(steps, areamqzm2, label="AREAMQZM2")
    ax.plot(steps, areamqzm3, label="AREAMQZM3")
    ax.legend()
    ax.grid(True)


def plot_steerer_history(ax, observations, before_reset=None):
    areamcvm1 = np.array([obs["magnets"][2] for obs in observations])
    areamchm2 = np.array([obs["magnets"][4] for obs in observations])

    if before_reset is not None:
        areamcvm1 = np.insert(areamcvm1, 0, before_reset[2])
        areamchm2 = np.insert(areamchm2, 0, before_reset[4])

    start = 0 if before_reset is None else -1
    steps = np.arange(start, len(observations))

    ax.set_title("Steerers")
    ax.set_xlabel("Step")
    ax.set_ylabel("Kick (mrad)")
    ax.set_xlim([start, len(observations) + 1])
    ax.plot(steps, areamcvm1 * 1e3, label="AREAMCVM1")
    ax.plot(steps, areamchm2 * 1e3, label="AREAMCHM2")
    ax.legend()
    ax.grid(True)


def plot_beam_history(ax, observations, before_reset=None):
    mu_x = np.array([obs["beam"][0] for obs in observations])
    sigma_x = np.array([obs["beam"][1] for obs in observations])
    mu_y = np.array([obs["beam"][2] for obs in observations])
    sigma_y = np.array([obs["beam"][3] for obs in observations])

    if before_reset is not None:
        mu_x = np.insert(mu_x, 0, before_reset[0])
        sigma_x = np.insert(sigma_x, 0, before_reset[1])
        mu_y = np.insert(mu_y, 0, before_reset[2])
        sigma_y = np.insert(sigma_y, 0, before_reset[3])

    target_beam = observations[0]["target"]

    start = 0 if before_reset is None else -1
    steps = np.arange(start, len(observations))

    ax.set_title("Beam Parameters")
    ax.set_xlim([start, len(observations) + 1])
    ax.set_xlabel("Step")
    ax.set_ylabel("(mm)")
    ax.plot(steps, mu_x * 1e3, label=r"$\mu_x$", c="tab:blue")
    ax.plot(steps, [target_beam[0] * 1e3] * len(steps), ls="--", c="tab:blue")
    ax.plot(steps, sigma_x * 1e3, label=r"$\sigma_x$", c="tab:orange")
    ax.plot(steps, [target_beam[1] * 1e3] * len(steps), ls="--", c="tab:orange")
    ax.plot(steps, mu_y * 1e3, label=r"$\mu_y$", c="tab:green")
    ax.plot(steps, [target_beam[2] * 1e3] * len(steps), ls="--", c="tab:green")
    ax.plot(steps, sigma_y * 1e3, label=r"$\sigma_y$", c="tab:red")
    ax.plot(steps, [target_beam[3] * 1e3] * len(steps), ls="--", c="tab:red")
    ax.legend()
    ax.grid(True)


def plot_beam_image(ax, img, screen_resolution, pixel_size, title="Beam Image"):
    screen_size = screen_resolution * pixel_size

    ax.set_title(title)
    ax.set_xlabel("(mm)")
    ax.set_ylabel("(mm)")
    ax.imshow(
        img,
        vmin=0,
        aspect="equal",
        interpolation="none",
        extent=(
            -screen_size[0] / 2 * 1e3,
            screen_size[0] / 2 * 1e3,
            -screen_size[1] / 2 * 1e3,
            screen_size[1] / 2 * 1e3,
        ),
    )


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


class TQDMWrapper(gym.Wrapper):
    """
    Uses TQDM to show a progress bar for every step taken by the environment. If the
    passed `env` is already wrapper in a `TimeLimit` wrapper, this wrapper will use that
    as the maximum number of steps for the progress bar.
    """

    def reset(self):
        if hasattr(self, "pbar"):
            self.pbar.close()

        obs = super().reset()

        if is_wrapped(self.env, TimeLimit):
            time_limit = unwrap_wrapper(self.env, TimeLimit)
            self.pbar = tqdm(total=time_limit._max_episode_steps)
        else:
            self.pbar = tqdm()

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.pbar.update()
        return obs, reward, done, info

    def close(self):
        if hasattr(self, "pbar"):
            self.pbar.close()

        super().close()
