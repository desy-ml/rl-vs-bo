"""Evaluate BO with a NN prior"""
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit

from bayesopt import (
    BeamNNPrior,
    get_new_bound,
    get_next_samples,
    observation_to_scaled_action,
)
from ea_train import ARESEACheetah
from trial import Trial, load_trials
from utils import FilterAction, RecordEpisode


def try_problem(
    trial_index: int,
    trial: Trial,
    folder_name: str = "bo_nnprior",
    use_nn_prior: bool = True,
    fit_weight: bool = False,
    acquisition: str = "EI",
):
    config = {
        "action_mode": "direct_unidirectional_quads",
        "filter_action": None,
        "filter_observation": None,  # ["beam", "magnets", "target"],
        "incoming_mode": "constant",
        "incoming_values": trial.incoming_beam,
        "magnet_init_mode": "constant",
        "magnet_init_values": np.array([10, -10, 0, 10, 0]),
        "max_steps": 150,
        "misalignment_mode": "constant",
        "misalignment_values": trial.misalignments,
        "rescale_action": (-1, 1),
        "reward_mode": "feedback",
        "target_beam_mode": "constant",
        "target_beam_values": trial.target_beam,
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
        "w_on_screen": 10.0,
        "w_sigma_x": 1.0,
        "w_sigma_x_in_threshold": 0.0,
        "w_sigma_y": 1.0,
        "w_sigma_y_in_threshold": 0.0,
        "w_time": 0.0,
        "acquisition": acquisition,
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
        log_beam_distance=True,
        normalize_beam_distance=False,
    )
    env = TimeLimit(env, config["max_steps"])
    env = RecordEpisode(env, save_dir=f"{folder_name}/problem_{trial_index:03d}")
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
    acquisition = config["acquisition"]
    init_x = config["init_x"]
    init_samples = config["init_samples"]

    # Actual optimisation
    observation = env.reset()
    done = False

    # Construct the NN prior for BO
    target_beam = torch.tensor(config["target_beam_values"])
    if use_nn_prior:
        nn_priormean = BeamNNPrior(
            target=target_beam, w_on_screen=config["w_on_screen"]
        )

        if not fit_weight:
            # Try without refitting the prior mean
            for param in nn_priormean.mlp.parameters():
                param.requires_grad = False
    else:
        nn_priormean = None  # Use default mean module for nn

    # Initialization
    x_dim = env.action_space.shape[0]
    # bounds = torch.tensor(
    #     np.array([env.action_space.low, env.action_space.high]), dtype=torch.float32
    # )
    if init_x is not None:  # From fix starting points
        X = torch.tensor(init_x.reshape(-1, x_dim), dtype=torch.float32)
    else:  # Random Initialization-5.7934
        action_i = observation_to_scaled_action(
            env, observation, config["filter_action"]
        )
        X = torch.tensor([action_i], dtype=torch.float32)
        bounds = get_new_bound(env, action_i, stepsize)
        for i in range(init_samples - 1):
            new_action = np.random.uniform(low=bounds[0], high=bounds[1]).reshape(1, -1)
            X = torch.cat([X, torch.tensor(new_action)])
    # Sample initial Ys to build GP
    Y = torch.empty((X.shape[0], 1))
    for i, action in enumerate(X):
        action = action.detach().numpy()
        _, objective, done, _ = env.step(action)
        Y[i] = torch.tensor(objective)

    # Actual BO Loop
    while not done:
        current_action = X[-1].detach().numpy()
        bounds = get_new_bound(env, current_action, stepsize)
        action_t = get_next_samples(
            X,
            Y.double(),
            Y.max(),
            torch.tensor(bounds, dtype=torch.float32),
            n_points=1,
            acquisition=acquisition,
            mean_module=nn_priormean,  # Use NN as prior mean
        )
        action = action_t.detach().numpy().flatten()
        _, objective, done, _ = env.step(action)

        # append data
        X = torch.cat([X, action_t])
        Y = torch.cat([Y, torch.tensor([[objective]], dtype=torch.float32)])

    # Set back to best values found
    set_to_best = True
    if set_to_best:
        action = X[Y.argmax()].detach().numpy()
        observation, reward, done, _ = env.step(action)

    env.close()


def main():
    trials = load_trials(Path("trials.yaml"))

    with ProcessPoolExecutor() as executor:
        print("Starting default prior with EI acq")
        executor.map(
            try_problem,
            range(len(trials)),
            trials,
            repeat("bo_noprior_ei"),
            repeat(False),
            repeat(False),
            repeat("EI"),
        )

    with ProcessPoolExecutor() as executor:
        print("Starting default prior with UCB acq")
        executor.map(
            try_problem,
            range(len(trials)),
            trials,
            repeat("bo_noprior_ucb"),
            repeat(False),
            repeat(False),
            repeat("UCB"),
        )

    with ProcessPoolExecutor() as executor:
        print("Starting default prior with PI acq")
        executor.map(
            try_problem,
            range(len(trials)),
            trials,
            repeat("bo_noprior_pi"),
            repeat(False),
            repeat(False),
            repeat("PI"),
        )

    with ProcessPoolExecutor() as executor:
        print("Starting NN no fit with EI acq")
        executor.map(
            try_problem,
            range(len(trials)),
            trials,
            repeat("bo_nnprior_ei"),
            repeat(True),
            repeat(False),
            repeat("EI"),
        )

    with ProcessPoolExecutor() as executor:
        print("Starting NN no fit with UCB acq")
        executor.map(
            try_problem,
            range(len(trials)),
            trials,
            repeat("bo_nnprior_ucb"),
            repeat(True),
            repeat(False),
            repeat("UCB"),
        )

    with ProcessPoolExecutor() as executor:
        print("Starting NN no fit with PI acq")
        executor.map(
            try_problem,
            range(len(trials)),
            trials,
            repeat("bo_nnprior_pi"),
            repeat(True),
            repeat(False),
            repeat("PI"),
        )

    # Testing
    # try_problem(
    #     0,
    #     problems[0],
    #     "bo_noprior_ucb",
    #     False,
    #     False,
    #     "UCB",
    # )
    # try_problem(
    #     0,
    #     problems[0],
    #     "bo_nnprior_ucb",
    #     False,
    #     acquisition="UCB",
    # )
    # try_problem(
    #     0,
    #     problems[0],
    #     "bo_nnprior_pi",
    #     False,
    #     acquisition="PI",
    # )
    # print("Finishing no fit")
    # try_problem(
    #     0,
    #     problems[0],
    #     # "bo_nnprior_nofit",
    #     # False,
    #     "bo_nnprior_fitweight",
    #     True,
    # )


if __name__ == "__main__":
    main()
