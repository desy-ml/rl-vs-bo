import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

# import dummypydoocs as pydoocs
import gym
import numpy as np
import pydoocs
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

from ea_train import ARESEA
from utils import (
    ARESEAeLog,
    FilterAction,
    NotVecNormalize,
    PolishedDonkeyCompatibility,
    RecordEpisode,
    SetUpstreamSteererAtStep,
    TQDMWrapper,
    load_config,
)


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
    data_log_dir=None,
    progress_bar=False,
    callback=None,
):
    """
    Optimise beam in ARES EA using a reinforcement learning agent.
    """
    config = load_config(f"models/{model_name}/config")

    # Load the model
    model = PPO.load(f"models/{model_name}/model")

    callback = setup_callback(callback)

    # Create the environment
    env = ARESEADOOCS(
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=config["magnet_init_values"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
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
    if progress_bar:
        env = TQDMWrapper(env)
    if callback is not None:
        env = OptimizeFunctionCallback(env, callback)
    if data_log_dir is not None:
        env = RecordEpisode(env, save_dir=data_log_dir)
    if logbook:
        env = ARESEAeLog(env, model_name=model_name)
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

    callback.env = env

    # Actual optimisation
    observation = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
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
    data_log_dir=None,
    progress_bar=False,
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

    callback = setup_callback(callback)

    # Create the environment
    env = ARESEADOOCS(
        action_mode="delta",
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        max_quad_delta=30 * 0.1,
        max_steerer_delta=6e-3 * 0.1,
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
    if progress_bar:
        env = TQDMWrapper(env)
    if callback is not None:
        env = OptimizeFunctionCallback(env, callback)
    if data_log_dir is not None:
        env = RecordEpisode(env, save_dir=data_log_dir)
    if logbook:
        env = ARESEAeLog(env, model_name=model_name)
    env = RecordVideo(env, f"recordings_real/{datetime.now():%Y%m%d%H%M}")
    env = FlattenObservation(env)
    env = PolishedDonkeyCompatibility(env)
    env = NotVecNormalize(env, f"models/{model_name}/vec_normalize.pkl")
    env = RescaleAction(env, -1, 1)

    # TODO temporary for experiment 20 December 2022
    env = SetUpstreamSteererAtStep(
        env, steps_to_trigger=40, steerer="ARLIMCHM1", mrad=-0.1518
    )

    callback.env = env

    # Actual optimisation
    observation = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
    env.close()


def optimize_async(*args, **kwargs):
    """Run `optimize without blocking."""
    executor = ThreadPoolExecutor(max_workers=1)
    # executor.submit(optimize, *args, **kwargs)
    kwargs["model_name"] = "polished-donkey-996"
    executor.submit(optimize_donkey, *args, **kwargs)


class ARESEADOOCS(ARESEA):
    def __init__(
        self,
        action_mode="direct",
        include_beam_image_in_info=True,
        log_beam_distance: bool = False,
        magnet_init_mode="zero",
        magnet_init_values=None,
        max_quad_delta: Optional[float] = None,
        max_steerer_delta: Optional[float] = None,
        normalize_beam_distance: bool = True,
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
            log_beam_distance=log_beam_distance,
            magnet_init_mode=magnet_init_mode,
            magnet_init_values=magnet_init_values,
            max_quad_delta=max_quad_delta,
            max_steerer_delta=max_steerer_delta,
            normalize_beam_distance=normalize_beam_distance,
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
        self.beam_parameter_compute_failed = {"x": False, "y": False}
        self.reset_accelerator_was_just_called = False

    def is_beam_on_screen(self):
        return not all(self.beam_parameter_compute_failed.values())

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

    def reset_accelerator(self):
        # If reset changes the magnet settings, record magnets and beam before reset
        if self.magnet_init_mode is None:
            return

        self.update_accelerator()

        self.magnets_before_reset = self.get_magnets()
        self.screen_before_reset = self.get_beam_image()
        self.beam_before_reset = self.get_beam_parameters()

        # In order to record a screen image right after the accelerator was reset, this
        # flag is set so that we know to record the image the next time
        # `update_accelerator` is called.
        self.reset_accelerator_was_just_called = True

    def update_accelerator(self):
        self.beam_image = self.capture_clean_beam_image()

        # Record the beam image just after reset (because there is no info on reset).
        # It will be included in `info` of the next step.
        if self.reset_accelerator_was_just_called:
            self.screen_after_reset = self.beam_image
            self.reset_accelerator_was_just_called = False

    def get_beam_parameters(self):
        img = self.get_beam_image()
        pixel_size = self.get_pixel_size()
        resolution = self.get_screen_resolution()

        parameters = {}
        for axis, direction in zip([0, 1], ["x", "y"]):
            projection = img.sum(axis=axis)
            minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
            filtered = uniform_filter1d(
                minfiltered, size=5, mode="nearest"
            )  # TODO rethink filters

            (half_values,) = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2

                # If (almost) all pixels are in FWHM, the beam might not be on screen
                self.beam_parameter_compute_failed[direction] = (
                    len(half_values) > 0.95 * resolution[axis]
                )
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
        Capture a clean image of the beam from the screen using `average` images with
        beam on and `average` images of the background and then removing the background.

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

        return flipped.astype(np.uint16)

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
        """
        Sets the bool switch of the cathode laser event to `setto` and waits a second.
        """
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)

    def get_accelerator_info(self):
        # If magnets or the beam were recorded before reset, add them info on the first
        # step, so a generalised data recording wrapper captures them.
        info = {}

        if hasattr(self, "magnets_before_reset"):
            info["magnets_before_reset"] = self.magnets_before_reset
            del self.magnets_before_reset
        if hasattr(self, "screen_before_reset"):
            info["screen_before_reset"] = self.screen_before_reset
            del self.screen_before_reset
        if hasattr(self, "beam_before_reset"):
            info["beam_before_reset"] = self.beam_before_reset
            del self.beam_before_reset

        if hasattr(self, "screen_after_reset"):
            info["screen_after_reset"] = self.screen_after_reset
            del self.screen_after_reset

        # Gain of camera for AREABSCR1
        info["camera_gain"] = pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/GAINRAW")[
            "data"
        ]

        # Steerers upstream of Experimental Area
        for steerer in ["ARLIMCHM1", "ARLIMCVM1", "ARLIMCHM2", "ARLIMCVM2"]:
            response = pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{steerer}/KICK_MRAD.RBV")
            info[steerer] = response["data"] / 1000

        # Gun solenoid
        info["gun_solenoid"] = pydoocs.read(
            "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/FIELD.RBV"
        )["data"]

        return info


def setup_callback(callback):
    """
    Prepare the callback for the actual optimisation run and return a callback that
    works exactly as expected.
    """
    if callback is None:
        callback = BaseCallback()
    elif isinstance(callback, list):
        callback = CallbackList(callback)
    return callback


class BaseCallback:
    """
    Base for callbacks to pass into `optimize` function and get information at different
    points of the optimisation.
    Provides access to the environment via `self.env`.
    """

    def __init__(self):
        self.env = None

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

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value
        for callback in self.callbacks:
            callback.env = self._env

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
        print("""environment_close""")


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
        done = done or self.callback.environment_step(obs, reward, done, info)
        return obs, reward, done, info

    def close(self):
        super().close()
        self.callback.environment_close()
