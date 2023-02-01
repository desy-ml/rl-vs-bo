from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import wandb
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from scipy.optimize import minimize
from stable_baselines3.common.env_util import unwrap_wrapper

from backend import EACheetahBackend
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import FilterAction, RecordEpisode


def try_problem(trial_index: int, trial: Trial, config: dict) -> float:
    config["incoming_values"] = trial.incoming_beam
    config["misalignment_values"] = trial.misalignments
    config["target_beam_values"] = trial.target_beam

    # Create the environment
    cheetah_backend = EACheetahBackend(
        incoming_mode=config["incoming_mode"],
        incoming_values=config["incoming_values"],
        misalignment_mode=config["misalignment_mode"],
        misalignment_values=config["misalignment_values"],
    )
    env = EATransverseTuning(
        backend=cheetah_backend,
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=config["magnet_init_values"],
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

    _ = env.reset()

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
                [
                    config["is00"],
                    config["is01"],
                    config["is02"],
                    config["is03"],
                    config["is04"],
                ],
                [
                    config["is10"],
                    config["is11"],
                    config["is12"],
                    config["is13"],
                    config["is14"],
                ],
                [
                    config["is20"],
                    config["is21"],
                    config["is22"],
                    config["is23"],
                    config["is24"],
                ],
                [
                    config["is30"],
                    config["is31"],
                    config["is32"],
                    config["is33"],
                    config["is34"],
                ],
                [
                    config["is40"],
                    config["is41"],
                    config["is42"],
                    config["is43"],
                    config["is44"],
                ],
                [
                    config["is50"],
                    config["is51"],
                    config["is52"],
                    config["is53"],
                    config["is54"],
                ],
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

    trials = load_trials(Path("trials.yaml"))

    configed_try_problem = partial(try_problem, config=config)

    with ProcessPoolExecutor() as executor:
        futures = executor.map(configed_try_problem, range(len(trials)), trials)
        maes = np.array(list(futures))

    wandb.log({"mae_mean": maes.mean(), "mae_std": maes.std()})


if __name__ == "__main__":
    main()
