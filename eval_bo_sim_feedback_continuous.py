from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gym.wrappers import RescaleAction, TimeLimit
from tqdm import tqdm

from backend import EACheetahBackend
from bayesopt import BayesianOptimizationAgent
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import AnimateIncomingBeam, RecordEpisode


def try_problem(trial_index: int, trial: Trial, next_trial: Trial) -> None:
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
    env = TimeLimit(env, 80)
    env = AnimateIncomingBeam(
        env, over_n_steps=80, to_beam_parameters=next_trial.incoming_beam
    )
    env = RecordEpisode(
        env,
        save_dir=(
            f"data/bo_vs_rl/simulation/bo_feedback_continuous/problem_{trial_index:03d}"
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
    next_trials = trials[1:] + [trials[0]]

    with ProcessPoolExecutor() as executor:
        _ = tqdm(
            executor.map(try_problem, range(len(trials)), trials, next_trials),
            total=300,
        )


if __name__ == "__main__":
    main()
