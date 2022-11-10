import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from tqdm.notebook import tqdm

from bayesopt import calculate_objective, get_new_bound, get_next_samples, scale_action
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
        "action_mode": "direct_unidirectional_quads",
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
        "obj_function": "logmae",
        "acquisition": "EI",
        "init_x": None,
        "init_samples": 5,
        "stepsize": 0.1,
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
    env = RecordEpisode(env, save_dir=f"bo_evaluation/problem_{problem_index:03d}")
    if config["filter_observation"] is not None:
        env = FilterObservation(env, config["filter_observation"])
    if config["filter_action"] is not None:
        env = FilterAction(env, config["filter_action"], replace=0)
    env = FlattenObservation(env)
    if config["rescale_action"] is not None:
        env = RescaleAction(
            env, config["rescale_action"][0], config["rescale_action"][1]
        )

    stepsize = config["stepsize"]
    obj_function = config["obj_function"]
    acquisition = config["acquisition"]
    init_x = config["init_x"]
    init_samples = config["init_samples"]

    # Actual optimisation
    observation = env.reset()
    done = False

    # Initialization
    x_dim = env.action_space.shape[0]
    # bounds = torch.tensor(
    #     np.array([env.action_space.low, env.action_space.high]), dtype=torch.float32
    # )
    if init_x is not None:  # From fix starting points
        X = torch.tensor(init_x.reshape(-1, x_dim), dtype=torch.float32)
    else:  # Random Initialization-5.7934
        action_i = scale_action(env, observation, config["filter_action"])
        X = torch.tensor([action_i], dtype=torch.float32)
        bounds = get_new_bound(env, action_i, stepsize)
        for i in range(init_samples - 1):
            new_action = np.random.uniform(low=bounds[0], high=bounds[1]).reshape(1, -1)
            X = torch.cat([X, torch.tensor(new_action)])
    # Sample initial Ys to build GP
    Y = torch.empty((X.shape[0], 1))
    for i, action in enumerate(X):
        action = action.detach().numpy()
        observation, reward, done, info = env.step(action)
        objective = calculate_objective(env, observation, reward, obj=obj_function)
        Y[i] = torch.tensor(objective)

    # Actual BO Loop
    while not done:
        current_action = X[-1].detach().numpy()
        bounds = get_new_bound(env, current_action, stepsize)
        action_t = get_next_samples(
            X,
            Y,
            Y.max(),
            torch.tensor(bounds, dtype=torch.float32),
            n_points=1,
            acquisition=acquisition,
        )
        action = action_t.detach().numpy().flatten()
        observation, reward, done, info = env.step(action)
        objective = calculate_objective(env, observation, reward, obj=obj_function)

        # append data
        X = torch.cat([X, action_t])
        Y = torch.cat([Y, torch.tensor([[objective]], dtype=torch.float32)])

    # Set back to
    set_to_best = True
    if set_to_best:
        action = X[Y.argmax()].detach().numpy()
        observation, reward, done, info = env.step(action)

    env.close()


def main():
    with open("problems.json", "r") as f:
        problems = json.load(f)

    with ProcessPoolExecutor() as executor:
        futures = tqdm(
            executor.map(try_problem, range(len(problems)), problems), total=300
        )


if __name__ == "__main__":
    main()
