import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
from gym.wrappers import RescaleAction, TimeLimit
from stable_baselines3.common.env_util import unwrap_wrapper
from tqdm import tqdm

from backend import EACheetahBackend
from bayesopt import BayesianOptimizationAgent
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import RecordEpisode


def try_problem(trial_index: int, trial: Trial):
    print(f" -> Trial {trial_index}")

    # Create the environment
    cheetah_backend = EACheetahBackend(
        incoming_mode="constant",
        incoming_values=trial.incoming_beam,
        misalignment_mode="constant",
        misalignment_values=trial.misalignments,
    )
    env = EATransverseTuning(
        backend=cheetah_backend,
        action_mode="direct_unidirectional_quads",
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        reward_mode="feedback",
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
        logarithmic_beam_distance=True,
        normalize_beam_distance=False,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(env)
    env = RescaleAction(env, -3, 3)

    model = BayesianOptimizationAgent(
        env=env,
        stepsize=0.1,
        init_samples=5,
        acquisition="EI",
        mean_module=None,
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

    # Get trials for SLURM task
    slurm_array_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    array_task_trials = modified_trials[
        slurm_array_task_id * 1_000 : slurm_array_task_id * 1_000 + 1_000
    ]

    # Run trials
    with ProcessPoolExecutor() as executor:
        futures = tqdm(
            executor.map(try_problem, range(len(array_task_trials)), array_task_trials),
            total=len(array_task_trials),
        )

        results = [
            {"target_beam": trial.target_beam, "final_beam": future}
            for trial, future in zip(array_task_trials, futures)
        ]

    # Save data
    Path.mkdir("data/bo_vs_rl/simulation/bo_grid/", parents=True, exist_ok=True)
    with open(
        f"data/bo_vs_rl/simulation/bo_grid/bo_{slurm_array_task_id:03d}.pkl", "wb"
    ) as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
