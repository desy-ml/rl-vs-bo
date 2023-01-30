from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gym.wrappers import RescaleAction, TimeLimit
from tqdm.notebook import tqdm

from bayesopt import BayesianOptimizationAgent
from ea_train import ARESEACheetah
from trial import Trial, load_trials
from utils import RecordEpisode


def try_problem(trial_index: int, trial: Trial):
    # Create the environment
    env = ARESEACheetah(
        action_mode="direct_unidirectional_quads",
        incoming_mode="constant",
        incoming_values=trial.incoming_beam,
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        misalignment_mode="constant",
        misalignment_values=trial.misalignments,
        reward_mode="feedback",
        target_beam_mode="constant",
        target_beam_values=trial.target_beam,
        target_mu_x_threshold=None,
        target_mu_y_threshold=None,
        target_sigma_x_threshold=None,
        target_sigma_y_threshold=None,
        threshold_hold=5,
        w_done=0.0,
        w_mu_x=1.0,
        w_mu_x_in_threshold=0.0,
        w_mu_y=1.0,
        w_mu_y_in_threshold=0.0,
        w_on_screen=10.0,
        w_sigma_x=1.0,
        w_sigma_x_in_threshold=0.0,
        w_sigma_y=1.0,
        w_sigma_y_in_threshold=0.0,
        w_time=0.0,
        log_beam_distance=True,
        normalize_beam_distance=False,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(
        env,
        save_dir=(
            f"data/bo_vs_rl/simulation/bo_refactor_test_8/problem_{trial_index:03d}"
        ),
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


def main():
    trials = load_trials(Path("trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()
