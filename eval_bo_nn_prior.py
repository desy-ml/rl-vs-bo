"""Evaluate BO with a NN prior"""
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
from gym.wrappers import RescaleAction, TimeLimit
from tqdm import tqdm

from backend import EACheetahBackend
from bayesopt import BayesianOptimizationAgent, BeamNNPrior
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import RecordEpisode


def try_problem(
    trial_index: int,
    trial: Trial,
    directory_name: str = "bo_nn_prior",
    use_nn_prior: bool = True,
    fit_weight: bool = False,
    acquisition: str = "EI",
):
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
    env = RecordEpisode(
        env,
        save_dir=f"data/bo_vs_rl/simulation/{directory_name}/problem_{trial_index:03d}",
    )
    env = RescaleAction(env, -3, 3)

    mean_module = None

    # Construct the NN prior for BO
    target_beam = torch.tensor(trial.target_beam)
    if use_nn_prior:
        mean_module = BeamNNPrior(target=target_beam, w_on_screen=10.0)

        if not fit_weight:
            # Try without refitting the prior mean
            for param in mean_module.mlp.parameters():
                param.requires_grad = False

    model = BayesianOptimizationAgent(
        env=env,
        stepsize=0.1,
        init_samples=5,
        acquisition=acquisition,
        mean_module=mean_module,
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
        print("Starting NN no fit with EI acq")
        _ = tqdm(
            executor.map(
                try_problem,
                range(len(trials)),
                trials,
                repeat("bo_nn_prior_2"),
                repeat(True),
                repeat(False),
                repeat("EI"),
            ),
            total=300,
        )


if __name__ == "__main__":
    main()
