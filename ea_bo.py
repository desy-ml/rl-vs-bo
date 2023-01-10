from datetime import datetime

import numpy as np
from gym.wrappers import (
    FilterObservation,
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)

from bayesopt import BayesianOptimizationAgent
from ea_optimize import (
    ARESEADOOCS,
    ARESEAeLog,
    BaseCallback,
    OptimizeFunctionCallback,
    TQDMWrapper,
    setup_callback,
)
from utils import FilterAction, RecordEpisode

config = {
    "action_mode": "direct_unidirectional_quads",
    "gamma": 0.99,
    # "filter_action": [0, 1, 3],
    "filter_action": None,
    "filter_observation": None,
    "frame_stack": None,
    "incoming_mode": "random",
    "incoming_values": None,
    "magnet_init_mode": "constant",
    # "magnet_init_values": np.array([10, -10, 0, 10, 0]),
    "misalignment_mode": "constant",
    "misalignment_values": np.zeros(8),
    "n_envs": 40,
    "normalize_observation": True,
    "normalize_reward": True,
    "rescale_action": (-3, 3),  # TODO this was -1, 1 in most real-world experiments
    "reward_mode": "feedback",
    "sb3_device": "auto",
    "target_beam_mode": "constant",
    "target_beam_values": np.zeros(4),
    "target_mu_x_threshold": 1e-5,
    "target_mu_y_threshold": 1e-5,
    "target_sigma_x_threshold": 1e-5,
    "target_sigma_y_threshold": 1e-5,
    "threshold_hold": 5,
    "time_limit": 50000,
    "vec_env": "subproc",
    "w_done": 0.0,
    "w_mu_x": 1.0,
    "w_mu_x_in_threshold": 0.0,
    "w_mu_y": 1.0,
    "w_mu_y_in_threshold": 0.0,
    "w_on_screen": 10.0,
    "w_sigma_x": 1.0,
    "w_sigma_x_in_threshold": 0.0,
    "w_sigma_y": 1.0,
    "w_sigma_y_in_threshold": 0.0,
    "w_time": 0.0,
}


# define a similar optimize function as in ea_optimize.py


def optimize(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=3.3198e-6,
    target_mu_y_threshold=3.3198e-6,
    target_sigma_x_threshold=3.3198e-6,
    target_sigma_y_threshold=3.3198e-6,
    max_steps=100,
    model_name="BO",
    logbook=False,
    data_log_dir=None,
    progress_bar=False,
    callback=BaseCallback(),
    stepsize=0.1,  # comparable to RL env
    acquisition="EI",
    init_x=None,
    init_samples=5,
    filter_action=None,
    magnet_init_values=np.array([10, -10, 0, 10, 0]),
    set_to_best=True,  # set back to best found setting after opt.
    mean_module=None,
):
    callback = setup_callback(callback)

    # Create the environment
    env = ARESEADOOCS(
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=magnet_init_values,
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
        log_beam_distance=True,
        normalize_beam_distance=False,
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

    model = BayesianOptimizationAgent(
        env=env,
        filter_action=filter_action,
        stepsize=stepsize,
        init_samples=init_samples,
        acquisition=acquisition,
        mean_module=mean_module,
    )

    callback.env = env

    # Actual optimisation
    observation = env.reset()
    reward = None
    done = False
    while not done:
        action = model.predict(observation, reward)
        observation, reward, done, info = env.step(action)

    # Set back to best
    if set_to_best:
        action = model.X[model.Y.argmax()].detach().numpy()
        env.step(action)

    env.close()
