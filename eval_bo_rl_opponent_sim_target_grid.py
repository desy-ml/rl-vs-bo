import pickle
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import unwrap_wrapper
from tqdm import tqdm

from backend import EACheetahBackend
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import NotVecNormalize, PolishedDonkeyCompatibility, RecordEpisode


def try_problem(trial_index: int, trial: Trial) -> None:
    model_name = "polar-lake-997"

    # Load the model
    model = TD3.load(f"models/{model_name}/model")

    # Create the environment
    cheetah_backend = EACheetahBackend(
        incoming_mode="constant",
        incoming_values=trial.incoming_beam,
        misalignment_mode="constant",
        misalignment_values=trial.misalignments,
    )
    env = EATransverseTuning(
        backend=cheetah_backend,
        action_mode="delta",
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        max_quad_delta=30 * 0.1,
        max_steerer_delta=6e-3 * 0.1,
        reward_mode="differential",
        target_beam_mode="constant",
        target_beam_values=trial.target_beam,
        target_mu_x_threshold=None,
        target_mu_y_threshold=None,
        target_sigma_x_threshold=None,
        target_sigma_y_threshold=None,
        threshold_hold=5,
        w_beam=1.0,
        w_mu_x=1.0,
        w_mu_y=1.0,
        w_on_screen=10.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(env)
    env = FilterObservation(env, ["beam", "magnets", "target"])
    env = FlattenObservation(env)
    env = PolishedDonkeyCompatibility(env)
    env = NotVecNormalize(env, f"models/{model_name}/vec_normalize.pkl")
    env = RescaleAction(env, -1, 1)

    # Actual optimisation
    observation = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)

    final_beam = unwrap_wrapper(env, RecordEpisode).observations[-1]["beam"]
    env.close()
    return final_beam


def main():
    original_trials = load_trials(Path("trials.yaml"))

    base_trial_index = 0

    base_trial = original_trials[base_trial_index]

    target_mu_xs = np.linspace(-2e-3, 2e-3, num=20)
    target_sigma_xs = np.geomspace(2e-5, 2e-3, num=20)
    target_mu_ys = np.linspace(-2e-3, 2e-3, num=20)
    target_sigma_ys = np.geomspace(2e-5, 2e-3, num=20)

    # Create trials
    modified_trials = []
    for base_trial_index, (mu_x, sigma_x, mu_y, sigma_y) in enumerate(
        product(target_mu_xs, target_sigma_xs, target_mu_ys, target_sigma_ys)
    ):
        trial = deepcopy(base_trial)
        trial.target_beam = np.array([mu_x, sigma_x, mu_y, sigma_y])
        modified_trials.append(trial)

    # Run trials
    with ProcessPoolExecutor() as executor:
        futures = tqdm(
            executor.map(try_problem, range(len(modified_trials)), modified_trials),
            total=len(modified_trials),
        )

        results = [
            {"target_beam": trial.target_beam, "final_beam": future}
            for trial, future in zip(modified_trials, futures)
        ]

    # Save data
    with open("data/bo_vs_rl/simulation/rl_grid/polar_lake.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
