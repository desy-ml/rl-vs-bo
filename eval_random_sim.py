from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from tqdm import tqdm

from backend import CheetahBackend
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import RecordEpisode


def try_problem(trial_index: int, trial: Trial) -> None:
    # Create the environment
    cheetah_backend = CheetahBackend(
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
    env = RecordEpisode(
        env, save_dir=f"data/bo_vs_rl/simulation/random_test/problem_{trial_index:03d}"
    )
    env = FilterObservation(env, ["beam", "magnets", "target"])
    env = FlattenObservation(env)
    env = RescaleAction(env, -1, 1)

    # Actual optimisation
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    env.close()


def main():
    trials = load_trials(Path("trials.yaml"))

    with ProcessPoolExecutor() as executor:
        futures = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)
        for future in futures:
            pass


if __name__ == "__main__":
    main()
