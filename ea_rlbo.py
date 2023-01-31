from datetime import datetime

import numpy as np
import torch
from gym import spaces
from gym.wrappers import FlattenObservation, RecordVideo, RescaleAction, TimeLimit
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import unwrap_wrapper

from bayesopt import BayesianOptimizationAgent, observation_to_scaled_action
from ea_optimize import ARESEADOOCS, OptimizeFunctionCallback, setup_callback
from utils import (
    ARESEAeLog,
    NotVecNormalize,
    PolishedDonkeyCompatibility,
    RecordEpisode,
    TQDMWrapper,
)


def optimize_donkey_bo_combo(
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
    rl_steps=10,
    bo_takeover=None,  # Set to MAE obove which BO takes over or to None for no takeover (e.g. 0.00015˚)
    stepsize=0.1,  # comparable to RL env
    acquisition="EI",
    beta=0.2,
    set_to_best=True,  # set back to best found setting after opt.
    mean_module=None,
):
    """
    Function used for optimisation during operation.

    Note: Current version only works for polished-donkey-996.
    """
    # config = read_from_yaml(f"models/{model}/config")
    assert (
        model_name == "polished-donkey-996"
    ), "Current version only works for polished-donkey-996."

    callback = setup_callback(callback)

    # Create the environment
    env = ARESEADOOCS(
        action_mode="delta",
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        max_quad_delta=30 * 0.1,
        max_steerer_delta=6e-3 * 0.1,
        reward_mode="feedback",
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
        w_on_screen=10.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
        log_beam_distance=True,
        normalize_beam_distance=False,
    )
    if max_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_steps)
    if progress_bar:
        env = TQDMWrapper(env)
    if callback is not None:
        env = OptimizeFunctionCallback(env, callback)
    env = (
        RecordEpisode(env, save_dir=data_log_dir)
        if data_log_dir is not None
        else RecordEpisode(env)
    )
    if logbook:
        env = ARESEAeLog(env, model_name=model_name)
    env = RecordVideo(env, f"recordings_real/{datetime.now():%Y%m%d%H%M}")
    env = FlattenObservation(env)
    env = PolishedDonkeyCompatibility(env)
    env = NotVecNormalize(env, f"models/{model_name}/vec_normalize.pkl")
    env = RescaleAction(env, -1, 1)

    # Load models
    rl_model = TD3.load(f"models/{model_name}/model")
    bo_model = BayesianOptimizationAgent(
        env=env,
        stepsize=0.05,
        init_samples=rl_steps,
        acquisition="UCB",
        mean_module=None,
        beta=0.01,
    )

    elog_wrapper = unwrap_wrapper(env, ARESEAeLog)

    callback.env = env

    observation = env.reset()
    done = False

    # RL agent's turn
    i = 0
    while i < rl_steps and not done:
        action, _ = rl_model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        i += 1

    if (
        bo_takeover is not None
        and unwrap_wrapper(env, RecordEpisode).infos[-1]["l1_distance"]
        > bo_takeover * 4
    ):
        print("BO is taking over")
        # Prepare env for BO
        env = unwrap_wrapper(env, RecordVideo)
        env.unwrapped.action_mode = "direct"  # TODO direct vs direct_unidirectional?
        env.unwrapped.action_space = spaces.Box(
            low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
            high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
        )
        env.unwrapped.threshold_hold = 1
        env = RescaleAction(env, -6, 6)  # Twice the size because bidirectional

        # Retreive past examples and them feed to BO
        record_episode = unwrap_wrapper(env, RecordEpisode)

        rl_magnet_history = [
            observation_to_scaled_action(env, obs)
            for obs in record_episode.observations[1:]
        ]
        next_rl_proposal, _ = rl_model.predict(observation, deterministic=True)
        bo_model.X = torch.tensor(np.stack(rl_magnet_history + [next_rl_proposal]))

        rl_magnets = [
            observation_to_scaled_action(env, obs)
            for obs in record_episode.observations[1:]
        ]
        bo_model.X = torch.tensor(np.stack(rl_magnets))

        rl_objectives = record_episode.rewards
        bo_model.Y = torch.tensor(rl_objectives[:-1]).reshape(-1, 1)
        reward = rl_objectives[-1]

        # BO's turn
        while not done:
            action = bo_model.predict(observation, reward)
            observation, reward, done, info = env.step(action)

        # Set back to
        action = bo_model.X[bo_model.Y.argmax()].detach().numpy()
        env.step(action)
    else:
        while not done:
            action, _ = rl_model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)

        elog_wrapper.model_name += (
            f" not taken over by BO after {rl_steps} steps if MAE > {bo_takeover}"
        )

    env.close()
