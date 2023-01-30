from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from tqdm.notebook import tqdm

from bayesopt import BayesianOptimizationAgent
from ea_train import ARESEACheetah
from trial import Trial, load_trials
from utils import FilterAction, RecordEpisode


def try_problem(trial_index: int, trial: Trial):
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
        "rescale_action": (-3, 3),
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
        "obj_function": "logmae",
        "acquisition": "EI",
        "init_x": None,
        "init_samples": 5,
        "stepsize": 0.1,
        "mean_module": None,
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
    env = RecordEpisode(
        env,
        save_dir=f"data/bo_vs_rl/simulation/bo_refactor_test_6_final/problem_{trial_index:03d}",
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

    model = BayesianOptimizationAgent(
        env=env,
        filter_action=config["filter_action"],
        stepsize=config["stepsize"],
        init_samples=config["init_samples"],
        acquisition=config["acquisition"],
        mean_module=config["mean_module"],
    )

    # Actual optimisation
    observation = env.reset()
    reward = None
    done = False
    while not done:
        action = model.predict(observation, reward)
        observation, reward, done, info = env.step(action)

    # Set back to best
    action = model.X[model.Y.argmax()].detach().numpy()
    env.step(action)

    env.close()


def main():
    trials = load_trials(Path("trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()
