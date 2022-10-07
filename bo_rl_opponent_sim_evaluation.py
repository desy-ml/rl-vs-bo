import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from stable_baselines3 import TD3
from tqdm.notebook import tqdm

from bo_sim_evaluation import (
    convert_incoming_from_problem,
    convert_misalignments_from_problem,
    convert_target_from_problem,
)
from ea_train import ARESEACheetah
from utils import NotVecNormalize, PolishedDonkeyCompatibility, RecordEpisode


def try_problem(problem_index: dict, problem: int) -> None:
    model_name = "polished-donkey-996"

    # Load the model
    model = TD3.load(f"models/{model_name}/model")

    # Create the environment
    env = ARESEACheetah(
        action_mode="delta",
        incoming_mode="constant",
        incoming_values=convert_incoming_from_problem(problem),
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        misalignment_mode="constant",
        misalignment_values=convert_misalignments_from_problem(problem),
        reward_mode="differential",
        target_beam_mode="constant",
        target_beam_values=convert_target_from_problem(problem),
        target_mu_x_threshold=3.3198e-6,
        target_mu_y_threshold=3.3198e-6,
        target_sigma_x_threshold=3.3198e-6,
        target_sigma_y_threshold=3.3198e-6,
        threshold_hold=5,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(
        env, save_dir=f"bo_rl_opponent_evaluation/problem_{problem_index:03d}"
    )
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
