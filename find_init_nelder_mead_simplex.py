import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import wandb
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from scipy.optimize import minimize
from stable_baselines3.common.env_util import unwrap_wrapper
from tqdm import tqdm

from ea_train import ARESEACheetah
from utils import FilterAction, RecordEpisode


def convert_incoming_from_problem(problem: dict) -> np.ndarray:
    return np.array(
        [
            problem["incoming"]["energy"],
            problem["incoming"]["mu_x"],
            problem["incoming"]["mu_xp"],
            problem["incoming"]["mu_y"],
            problem["incoming"]["mu_yp"],
            problem["incoming"]["sigma_x"],
            problem["incoming"]["sigma_xp"],
            problem["incoming"]["sigma_y"],
            problem["incoming"]["sigma_yp"],
            problem["incoming"]["sigma_s"],
            problem["incoming"]["sigma_p"],
        ]
    )


def convert_misalignments_from_problem(problem: dict) -> np.ndarray:
    return np.array(problem["misalignments"])


def convert_target_from_problem(problem: dict) -> np.ndarray:
    return np.array(
        [
            problem["desired"][0],
            problem["desired"][2],
            problem["desired"][1],
            problem["desired"][3],
        ]
    )


def try_problem(problem_index: int, problem: dict, config: dict) -> float:
    config["incoming_values"] = convert_incoming_from_problem(problem)
    config["misalignment_values"] = convert_misalignments_from_problem(problem)
    config["target_beam_values"] = convert_target_from_problem(problem)

    # Create the environment
    env = ARESEACheetah(
        action_mode=config["action_mode"],
        incoming_mode=config["incoming_mode"],
        incoming_values=config["incoming_values"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=config["magnet_init_values"],
        misalignment_mode=config["misalignment_mode"],
        misalignment_values=config["misalignment_values"],
        reward_mode=config["reward_mode"],
        target_beam_mode=config["target_beam_mode"],
        target_beam_values=config["target_beam_values"],
        target_mu_x_threshold=config["target_mu_x_threshold"],
        target_mu_y_threshold=config["target_mu_y_threshold"],
        target_sigma_x_threshold=config["target_sigma_x_threshold"],
        target_sigma_y_threshold=config["target_sigma_y_threshold"],
        threshold_hold=config["threshold_hold"],
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
    env = TimeLimit(env, config["max_steps"])
    env = RecordEpisode(
        env,
        save_dir=None,
    )
    if config["filter_observation"] is not None:
        env = FilterObservation(env, config["filter_observation"])
    if config["filter_action"] is not None:
        env = FilterAction(env, config["filter_action"], replace=0)
    env = FlattenObservation(env)
    if config["rescale_action"] is not None:
        env = RescaleAction(
            env, config["rescale_action"][0], config["rescale_action"][1]
        )

    observation = env.reset()

    def objective_fun(magnets: np.ndarray) -> float:
        _, reward, _, _ = env.step(magnets)
        return -reward

    minimize(
        objective_fun,
        method="Nelder-Mead",
        x0=[0.1388888889, -0.1388888889, 0, 0.1388888889, 0],
        bounds=[(-1, 1)] * 5,
        options={
            "maxfev": 150,
            "initial_simplex": [
                [0, 0, 0, 0, 0],
                [0.1388888889, -0.1388888889, 0, -0.1388888889, 0],
                [-0.1388888889, 0.1388888889, 0, -0.1388888889, 0],
                [0, 0, 0.5, 0, -0.5],
                [-0.1388888889, -0.1388888889, 0, 0.1388888889, 0],
                [0, 0, -0.5, 0, 0.5],
            ],
        },
    )

    record_episode = unwrap_wrapper(env, RecordEpisode)
    mae = record_episode.infos[-1]["l1_distance"]

    env.close()

    return mae


def main():
    config = {
        "action_mode": "direct",
        "filter_action": None,
        "filter_observation": None,  # ["beam", "magnets", "target"],
        "incoming_mode": "constant",
        # "incoming_values": convert_incoming_from_problem(problem),
        "magnet_init_mode": "constant",
        "magnet_init_values": np.array([10, -10, 0, 10, 0]),
        "max_steps": 150,
        "misalignment_mode": "constant",
        # "misalignment_values": convert_misalignments_from_problem(problem),
        "rescale_action": (-1, 1),
        "reward_mode": "differential",
        "target_beam_mode": "constant",
        # "target_beam_values": convert_target_from_problem(problem),
        "target_mu_x_threshold": None,
        "target_mu_y_threshold": None,
        "target_sigma_x_threshold": None,
        "target_sigma_y_threshold": None,
        "threshold_hold": 5,
        "w_done": 0.0,
        "w_mu_x": 1.0,
        "w_mu_x_in_threshold": 0.0,
        "w_mu_y": 1.0,
        "w_mu_y_in_threshold": 0.0,
        "w_on_screen": 100.0,
        "w_sigma_x": 1.0,
        "w_sigma_x_in_threshold": 0.0,
        "w_sigma_y": 1.0,
        "w_sigma_y_in_threshold": 0.0,
        "w_time": 0.0,
        "is00": 0.0,
        "is01": 0.0,
        "is02": 0.0,
        "is03": 0.0,
        "is04": 0.0,
        "is10": 0.1388888889,
        "is11": -0.1388888889,
        "is12": 0.0,
        "is13": -0.1388888889,
        "is14": 0.0,
        "is20": -0.1388888889,
        "is21": 0.1388888889,
        "is22": 0.0,
        "is23": -0.1388888889,
        "is24": 0.0,
        "is30": 0.0,
        "is31": 0.0,
        "is32": 0.5,
        "is33": 0.0,
        "is34": -0.5,
        "is40": -0.1388888889,
        "is41": -0.1388888889,
        "is42": 0.0,
        "is43": 0.1388888889,
        "is44": 0.0,
        "is50": 0.0,
        "is51": 0.0,
        "is52": -0.5,
        "is53": 0.0,
        "is54": 0.5,
    }

    wandb.init(
        project="ares-ea-nelder-mead-init-simplex",
        entity="msk-ipc",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
        dir=".wandb",
    )
    config = dict(wandb.config)

    with open("problems.json", "r") as f:
        problems = json.load(f)

    configed_try_problem = partial(try_problem, config=config)

    with ProcessPoolExecutor() as executor:
        # futures = tqdm(
        #     executor.map(configed_try_problem, range(len(problems)), problems),
        #     total=300,
        # )
        futures = executor.map(configed_try_problem, range(len(problems)), problems)
        maes = np.array(list(futures))

    wandb.log({"mae_mean": maes.mean(), "mae_std": maes.std()})


if __name__ == "__main__":
    main()
