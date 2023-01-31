from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# import dummypydoocs as pydoocs
import gym
import numpy as np
from gym.wrappers import (
    FilterObservation,
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3 import PPO, TD3

from backend import DOOCSBackend
from ea_train import EATransverseTuning
from utils import (
    ARESEAeLog,
    FilterAction,
    NotVecNormalize,
    PolishedDonkeyCompatibility,
    RecordEpisode,
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
    env = DOOCSBackend(
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
    env = EATransverseTuning(
        backend=DOOCSBackend(),
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
        w_beam=1.0,
        w_mu_x=1.0,
        w_mu_y=1.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
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
