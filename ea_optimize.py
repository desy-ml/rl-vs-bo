import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO

# import pydoocs
import dummypydoocs as pydoocs
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import (
    FilterObservation,
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_util import unwrap_wrapper

from ea_train import ARESEA, read_from_yaml
from utils import (
    FilterAction,
    NotVecNormalize,
    PolishedDonkeyCompatibility,
    RecordEpisode,
    send_to_elog,
)

# import pydoocs

import dummypydoocs as pydoocs


def optimize(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=3.3198e-6,
    target_mu_y_threshold=3.3198e-6,
    target_sigma_x_threshold=3.3198e-6,
    target_sigma_y_threshold=3.3198e-6,
    max_steps=50,
    model_name="chocolate-totem-247",
    logbook=False,
    callback=None,
):
    """
    Optimise beam in ARES EA using a reinforcement learning agent.
    """
    config = read_from_yaml(f"models/{model_name}/config")

    # Load the model
    model = PPO.load(f"models/{model_name}/model")

    callback = CallbackList(callback) if isinstance(callback, list) else callback

    # Create the environment
    env = ARESEADOOCS(
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=config["magnet_init_values"],
        reward_mode=config["reward_mode"],
        target_beam_mode=config["target_beam_mode"],
        target_beam_values=np.array(
            [target_mu_x, target_sigma_x, target_mu_y, target_sigma_y]
        ),
        target_mu_x_threshold=target_mu_x_threshold,
        target_mu_y_threshold=target_mu_y_threshold,
        target_sigma_x_threshold=target_sigma_x_threshold,
        target_sigma_y_threshold=target_sigma_y_threshold,
        threshold_hold=1,
        w_done=config["w_done"],
        w_mu_x=config["w_mu_x"],
        w_mu_x_in_threshold=config["w_mu_x_in_threshold"],
        w_mu_y=config["w_mu_y"],
        w_mu_y_in_threshold=config["w_mu_y_in_threshold"],
        w_on_screen=config["w_on_screen"],
        w_sigma_x=config["w_sigma_x"],
        w_sigma_x_in_threshold=config["w_sigma_x_in_threshold"],
        w_sigma_y=config["w_sigma_y"],
        w_sigma_y_in_threshold=config["w_sigma_y_in_threshold"],
        w_time=config["w_time"],
    )
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    if callback is not None:
        env = OptimizeFunctionCallback(env, callback)
    env = RecordEpisode(env)
    if config["filter_observation"] is not None:
        env = FilterObservation(env, config["filter_observation"])
    if config["filter_action"] is not None:
        env = FilterAction(env, config["filter_action"], replace=0)
    env = FlattenObservation(env)
    if config["frame_stack"] is not None:
        env = FrameStack(env, config["frame_stack"])
    if config["rescale_action"] is not None:
        env = RescaleAction(
            env, config["rescale_action"][0], config["rescale_action"][1]
        )
    env = RecordVideo(env, video_folder=f"recordings_real/{datetime.now():%Y%m%d%H%M}")
    env = NotVecNormalize(env, f"models/{model_name}/vec_normalize.pkl")

    # Actual optimisation
    t_start = datetime.now()
    observation = env.reset()
    beam_image_before = env.get_beam_image()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
    t_end = datetime.now()

    recording = unwrap_wrapper(env, RecordEpisode)
    if logbook:
        report_ea_optimization_to_logbook(
            model_name,
            t_start,
            t_end,
            recording.observations,
            recording.infos,
            beam_image_before,
            target_mu_x_threshold,
            target_sigma_x_threshold,
            target_mu_y_threshold,
            target_sigma_y_threshold,
        )

    env.close()


def optimize_donkey(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=3.3198e-6,
    target_mu_y_threshold=3.3198e-6,
    target_sigma_x_threshold=3.3198e-6,
    target_sigma_y_threshold=3.3198e-6,
    max_steps=50,
    model_name="polished-donkey-996",
    logbook=False,
    callback=None,
):
    """
    Function used for optimisation during operation.

    Note: Current version only works for polished-donkey-996.
    """
    # config = read_from_yaml(f"models/{model}/config")
    assert (
        model_name == "polished-donkey-996"
    ), "Current version only works for polished-donkey-996."

    # Load the model
    model = TD3.load(f"models/{model_name}/model")

    # Create the environment
    env = ARESEADOOCS(
        action_mode="delta",
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        reward_mode="differential",
        target_beam_mode="constant",
        target_beam_values=np.array(
            [target_mu_x, target_sigma_x, target_mu_y, target_sigma_y]
        ),
        target_mu_x_threshold=target_mu_x_threshold,
        target_mu_y_threshold=target_mu_y_threshold,
        target_sigma_x_threshold=target_sigma_x_threshold,
        target_sigma_y_threshold=target_sigma_y_threshold,
    )
    if max_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_steps)
    env = RecordEpisode(env)
    env = RecordVideo(env, f"recordings_real/{datetime.now():%Y%m%d%H%M}")
    env = FlattenObservation(env)
    env = PolishedDonkeyCompatibility(env)
    env = NotVecNormalize(env, f"models/{model_name}/vec_normalize.pkl")
    env = RescaleAction(env, -1, 1)

    # Actual optimisation
    t_start = datetime.now()
    observation = env.reset()
    beam_image_before = env.get_beam_image()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
    t_end = datetime.now()

    recording = unwrap_wrapper(env, RecordEpisode)
    if logbook:
        report_ea_optimization_to_logbook(
            model_name,
            t_start,
            t_end,
            recording.observations,
            recording.infos,
            beam_image_before,
            target_mu_x_threshold,
            target_sigma_x_threshold,
            target_mu_y_threshold,
            target_sigma_y_threshold,
        )

    env.close()


def optimize_async(*args, **kwargs):
    """Run `optimize without blocking."""
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(optimize, *args, **kwargs)


class ARESEADOOCS(ARESEA):
    def __init__(
        self,
        action_mode="direct",
        include_beam_image_in_info=True,
        magnet_init_mode="zero",
        magnet_init_values=None,
        reward_mode="differential",
        target_beam_mode="random",
        target_beam_values=None,
        target_mu_x_threshold=3.3198e-6,
        target_mu_y_threshold=2.4469e-6,
        target_sigma_x_threshold=3.3198e-6,
        target_sigma_y_threshold=2.4469e-6,
        threshold_hold=1,
        w_done=1.0,
        w_mu_x=1.0,
        w_mu_x_in_threshold=1.0,
        w_mu_y=1.0,
        w_mu_y_in_threshold=1.0,
        w_on_screen=1.0,
        w_sigma_x=1.0,
        w_sigma_x_in_threshold=1.0,
        w_sigma_y=1.0,
        w_sigma_y_in_threshold=1.0,
        w_time=1.0,
    ):
        super().__init__(
            action_mode=action_mode,
            include_beam_image_in_info=include_beam_image_in_info,
            magnet_init_mode=magnet_init_mode,
            magnet_init_values=magnet_init_values,
            reward_mode=reward_mode,
            target_beam_mode=target_beam_mode,
            target_beam_values=target_beam_values,
            target_mu_x_threshold=target_mu_x_threshold,
            target_mu_y_threshold=target_mu_y_threshold,
            target_sigma_x_threshold=target_sigma_x_threshold,
            target_sigma_y_threshold=target_sigma_y_threshold,
            threshold_hold=threshold_hold,
            w_done=w_done,
            w_mu_x=w_mu_x,
            w_mu_x_in_threshold=w_mu_x_in_threshold,
            w_mu_y=w_mu_y,
            w_mu_y_in_threshold=w_mu_y_in_threshold,
            w_on_screen=w_on_screen,
            w_sigma_x=w_sigma_x,
            w_sigma_x_in_threshold=w_sigma_x_in_threshold,
            w_sigma_y=w_sigma_y,
            w_sigma_y_in_threshold=w_sigma_y_in_threshold,
            w_time=w_time,
        )

    def is_beam_on_screen(self):
        return True  # TODO find better logic

    def get_magnets(self):
        return np.array(
            [
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.RBV")["data"],
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.RBV")["data"],
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.RBV")["data"]
                / 1000,
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.RBV")["data"],
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.RBV")["data"]
                / 1000,
            ]
        )

    def set_magnets(self, magnets):
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.SP", magnets[0])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.SP", magnets[1])
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.SP", magnets[2] * 1000
        )
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.SP", magnets[3])
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.SP", magnets[4] * 1000
        )

        # Wait until magnets have reached their setpoints

        time.sleep(3.0)  # Wait for magnets to realise they received a command

        magnets = ["AREAMQZM1", "AREAMQZM2", "AREAMCVM1", "AREAMQZM3", "AREAMCHM1"]

        are_busy = [True] * 5
        are_ps_on = [True] * 5
        while any(are_busy) or not all(are_ps_on):
            are_busy = [
                pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/BUSY")["data"]
                for magnet in magnets
            ]
            are_ps_on = [
                pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/PS_ON")["data"]
                for magnet in magnets
            ]

    def update_accelerator(self):
        self.beam_image = self.capture_clean_beam_image()

    def get_beam_parameters(self):
        img = self.get_beam_image()
        pixel_size = self.get_pixel_size()

        parameters = {}
        for axis, direction in zip([0, 1], ["x", "y"]):
            projection = img.sum(axis=axis)
            minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
            filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

            (half_values,) = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2
            else:
                fwhm_pixel = 42  # TODO figure out what to do with these
                center_pixel = 42

            parameters[f"mu_{direction}"] = (
                center_pixel - len(filtered) / 2
            ) * pixel_size[axis]
            parameters[f"sigma_{direction}"] = fwhm_pixel / 2.355 * pixel_size[axis]

        parameters["mu_y"] = -parameters["mu_y"]

        return np.array(
            [
                parameters["mu_x"],
                parameters["sigma_x"],
                parameters["mu_y"],
                parameters["sigma_y"],
            ]
        )

    def get_beam_image(self):
        return self.beam_image

    def get_binning(self):
        return np.array(
            (
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL")[
                    "data"
                ],
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL")[
                    "data"
                ],
            )
        )

    def get_screen_resolution(self):
        return np.array(
            [
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/WIDTH")["data"],
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/HEIGHT")["data"],
            ]
        )

    def get_pixel_size(self):
        return (
            np.array(
                [
                    abs(
                        pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/X.POLY_SCALE")[
                            "data"
                        ][2]
                    )
                    / 1000,
                    abs(
                        pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/Y.POLY_SCALE")[
                            "data"
                        ][2]
                    )
                    / 1000,
                ]
            )
            * self.get_binning()
        )

    def capture_clean_beam_image(self, average=5):
        """
        Capture a clean image of the beam from the screen using `average` images with beam on and
        `average` images of the background and then removing the background.

        Saves the image to a property of the object.
        """
        # Laser off
        self.set_cathode_laser(False)
        background_images = self.capture_interval(n=average, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        self.set_cathode_laser(True)
        beam_images = self.capture_interval(n=average, dt=0.1)
        median_beam = np.median(beam_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16 - 1)
        flipped = np.flipud(removed)

        return flipped

    def capture_interval(self, n, dt):
        """Capture `n` images from the screen and wait `dt` seconds in between them."""
        images = []
        for _ in range(n):
            images.append(self.capture_screen())
            time.sleep(dt)
        return np.array(images)

    def capture_screen(self):
        """Capture and image from the screen."""
        return pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ")["data"]

    def set_cathode_laser(self, setto):
        """Sets the bool switch of the cathode laser event to `setto` and waits a second."""
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)


def report_ea_optimization_to_logbook(
    model_name,
    t_start,
    t_end,
    observations,
    infos,
    beam_image_before,
    target_mu_x_threshold,
    target_sigma_x_threshold,
    target_mu_y_threshold,
    target_sigma_y_threshold,
):
    # Create text message
    beam_before = observations[0]["beam"]
    beam_after = observations[-1]["beam"]
    target_beam = observations[0]["target"]
    target_threshold = np.array(
        [
            target_mu_x_threshold,
            target_sigma_x_threshold,
            target_mu_y_threshold,
            target_sigma_y_threshold,
        ]
    )
    final_magnets = observations[-1]["magnets"]
    steps_taken = len(observations) - 1
    success = np.abs(beam_after - target_beam) < target_threshold

    msg = f"""Reinforcement learning agent optimised beam on AREABSCR1

Agent: {model_name}
Start time: {t_start}
Time taken: {t_end - t_start}
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

    # Create plot as jpg
    fig, axs = plt.subplots(1, 5, figsize=(30, 4))
    plot_quadrupole_history(axs[0], observations)
    plot_steerer_history(axs[1], observations)
    plot_beam_history(axs[2], observations)
    plot_beam_image(
        axs[3],
        beam_image_before,
        screen_resolution=infos[0]["screen_resolution"],
        pixel_size=infos[0]["pixel_size"],
        title="Beam at Reset (Background Removed)",
    )
    plot_beam_image(
        axs[4],
        infos[-1]["beam_image"],
        screen_resolution=infos[-1]["screen_resolution"],
        pixel_size=infos[-1]["pixel_size"],
        title="Beam After (Background Removed)",
    )
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, dpi=300, format="jpg")
    buf.seek(0)
    img = bytes(buf.read())

    # Send to logbook
    send_to_elog(
        elog="areslog",
        author="Autonomous ARES",
        title="RL-based Beam Optimisation on AREABSCR1",
        severity="NONE",
        text=msg,
        image=img,
    )


def plot_quadrupole_history(ax, observations):
    areamqzm1 = [obs["magnets"][0] for obs in observations]
    areamqzm2 = [obs["magnets"][1] for obs in observations]
    areamqzm3 = [obs["magnets"][3] for obs in observations]

    steps = np.arange(len(observations))

    ax.set_title("Quadrupoles")
    ax.set_xlim([0, len(steps)])
    ax.set_xlabel("Step")
    ax.set_ylabel("Strength (1/m^2)")
    ax.plot(steps, areamqzm1, label="AREAMQZM1")
    ax.plot(steps, areamqzm2, label="AREAMQZM2")
    ax.plot(steps, areamqzm3, label="AREAMQZM3")
    ax.legend()
    ax.grid(True)


def plot_steerer_history(ax, observations):
    areamcvm1 = np.array([obs["magnets"][2] for obs in observations])
    areamchm2 = np.array([obs["magnets"][4] for obs in observations])

    steps = np.arange(len(observations))

    ax.set_title("Steerers")
    ax.set_xlabel("Step")
    ax.set_ylabel("Kick (mrad)")
    ax.set_xlim([0, len(steps)])
    ax.plot(steps, areamcvm1 * 1e3, label="AREAMCVM1")
    ax.plot(steps, areamchm2 * 1e3, label="AREAMCHM2")
    ax.legend()
    ax.grid(True)


def plot_beam_history(ax, observations):
    mu_x = np.array([obs["beam"][0] for obs in observations])
    sigma_x = np.array([obs["beam"][1] for obs in observations])
    mu_y = np.array([obs["beam"][2] for obs in observations])
    sigma_y = np.array([obs["beam"][3] for obs in observations])

    target_beam = observations[0]["target"]

    steps = np.arange(len(observations))

    ax.set_title("Beam Parameters")
    ax.set_xlim([0, len(steps)])
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


class BaseCallback:
    """
    Base for callbacks to pass into `optimize` function and get information at different
    points of the optimisation.
    """

    def environment_reset(self, obs):
        """Called after the environment's `reset` method has been called."""
        pass

    def environment_step(self, obs, reward, done, info):
        """
        Called after every call to the environment's `step` function.
        Return `True` tostop optimisation.
        """
        return False

    def environment_close(self):
        """Called after the optimization was finished."""
        pass


class CallbackList(BaseCallback):
    """Combines multiple callbacks into one."""

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def environment_reset(self, obs):
        for callback in self.callbacks:
            callback.environment_reset(obs)

    def environment_step(self, obs, reward, done, info):
        return any(
            [
                callback.environment_step(obs, reward, done, info)
                for callback in self.callbacks
            ]
        )

    def environment_close(self):
        for callback in self.callbacks:
            callback.environment_close()


class TestCallback(BaseCallback):
    """
    Very simple callback for testing. Prints method name and arguments whenever callback
    is called.
    """

    def environment_reset(self, obs):
        print(
            f"""environment_reset
    -> {obs = }"""
        )

    def environment_step(self, obs, reward, done, info):
        print(
            f"""environment_step
    -> {obs = }
    -> {reward = }
    -> {done = }
    -> {info = }"""
        )
        return False

    def environment_close(self):
        print(f"""environment_close""")


class OptimizeFunctionCallback(gym.Wrapper):
    """Wrapper to send screen image, beam parameters and optimisation end to GUI."""

    def __init__(self, env, callback):
        super().__init__(env)
        self.callback = callback

    def reset(self):
        obs = super().reset()
        self.callback.environment_reset(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.callback.environment_step(obs, reward, done, info)
        return obs, reward, done, info

    def close(self):
        super().close()
        self.callback.environment_close()
