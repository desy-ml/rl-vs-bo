from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gym.wrappers import RescaleAction, TimeLimit
from scipy.stats import linregress
from tqdm import tqdm

from backend import EACheetahBackend
from bayesopt import BayesianOptimizationAgent
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import RecordEpisode


def try_problem(trial_index: int, trial: Trial):
    # Create the environment
    cheetah_backend = EACheetahBackend(
        incoming_mode="constant",
        incoming_values=trial.incoming_beam,
        misalignment_mode="constant",
        misalignment_values=trial.misalignments,
    )
    env = EATransverseTuning(
        backend=cheetah_backend,
        action_mode="direct",
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
        unidirectional_quads=True,
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
    env = RecordEpisode(
        env,
        save_dir=f"data/bo_vs_rl/simulation/bo_quad_aligned/problem_{trial_index:03d}",
    )
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

    env.close()


def find_quad_aligned_incoming_beam_parameters(trial: Trial) -> np.ndarray:
    """
    Return incoming beam parameters that aligned the beam as best as possible to the
    misaligned quadrupoles.
    """
    s = [0.24, 0.79, 1.33]  # Hardcoded s-positions of centers of the quadrupoles
    x = trial.misalignments[[0, 2, 4]]
    y = trial.misalignments[[1, 3, 5]]

    x_fit = linregress(s, x)
    y_fit = linregress(s, y)

    incoming_beam = trial.incoming_beam.copy()
    incoming_beam[1] = x_fit.intercept
    incoming_beam[2] = np.arctan(x_fit.slope)
    incoming_beam[3] = y_fit.intercept
    incoming_beam[4] = np.arctan(y_fit.slope)

    return incoming_beam


def main():
    trials = load_trials(Path("trials.yaml"))

    for trial in trials:
        trial.incoming_beam = find_quad_aligned_incoming_beam_parameters(trial)

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()
