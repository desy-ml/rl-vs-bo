import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from scipy.optimize import minimize
from tqdm.notebook import tqdm

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


def try_problem(problem_index: int, problem: dict):
    config = {
        "action_mode": "direct",
        "filter_action": None,
        "filter_observation": None,  # ["beam", "magnets", "target"],
        "incoming_mode": "constant",
        "incoming_values": convert_incoming_from_problem(problem),
        "magnet_init_mode": "constant",
        "magnet_init_values": np.array([10, -10, 0, 10, 0]),
        "max_steps": 150,
        "misalignment_mode": "constant",
        "misalignment_values": convert_misalignments_from_problem(problem),
        "rescale_action": (-1, 1),
        "reward_mode": "differential",
        "target_beam_mode": "constant",
        "target_beam_values": convert_target_from_problem(problem),
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
    }

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
        save_dir=f"data/bo_vs_rl/simulation/nelder-mead/problem_{problem_index:03d}",
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

    env.close()


def main():
    with open("problems.json", "r") as f:
        problems = json.load(f)

    try_problem(0, problems[0])

    with ProcessPoolExecutor() as executor:
        futures = tqdm(
            executor.map(try_problem, range(len(problems)), problems), total=300
        )


if __name__ == "__main__":
    main()
