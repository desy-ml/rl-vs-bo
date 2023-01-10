import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from tqdm.notebook import tqdm

from ea_train import ARESEACheetah
from eval_bo_sim import (
    convert_incoming_from_problem,
    convert_misalignments_from_problem,
    convert_target_from_problem,
)
from utils import RecordEpisode


def try_problem(problem_index: dict, problem: int) -> None:
    # Create the environment
    env = ARESEACheetah(
        action_mode="direct",
        incoming_mode="constant",
        incoming_values=convert_incoming_from_problem(problem),
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        misalignment_mode="constant",
        misalignment_values=convert_misalignments_from_problem(problem),
        reward_mode="differential",
        target_beam_mode="constant",
        target_beam_values=convert_target_from_problem(problem),
        target_mu_x_threshold=None,
        target_mu_y_threshold=None,
        target_sigma_x_threshold=None,
        target_sigma_y_threshold=None,
        threshold_hold=5,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(
        env, save_dir=f"data/bo_vs_rl/simulation/random/problem_{problem_index:03d}"
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
    with open("problems.json", "r") as f:
        problems = json.load(f)

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(problems)), problems), total=300)


if __name__ == "__main__":
    main()
