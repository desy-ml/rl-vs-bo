from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from tqdm.notebook import tqdm

from ea_train import ARESEACheetah
from trial import Trial, load_trials
from utils import RecordEpisode


def try_problem(trial_index: int, trial: Trial) -> None:
    # Create the environment
    env = ARESEACheetah(
        action_mode="direct",
        incoming_mode="constant",
        incoming_values=trial.incoming_beam,
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        misalignment_mode="constant",
        misalignment_values=trial.misalignments,
        reward_mode="differential",
        target_beam_mode="constant",
        target_beam_values=trial.target_beam,
        target_mu_x_threshold=None,
        target_mu_y_threshold=None,
        target_sigma_x_threshold=None,
        target_sigma_y_threshold=None,
        threshold_hold=5,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(
        env, save_dir=f"data/bo_vs_rl/simulation/random/problem_{trial_index:03d}"
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
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()
