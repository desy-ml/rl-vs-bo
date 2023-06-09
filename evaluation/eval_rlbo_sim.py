from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from gym import spaces
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import unwrap_wrapper
from tqdm.notebook import tqdm

from backend import EACheetahBackend
from bayesopt import BayesianOptimizationAgent, observation_to_scaled_action
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import NotVecNormalize, PolishedDonkeyCompatibility, RecordEpisode


def try_problem(trial_index: int, trial: Trial) -> None:
    model_name = "polished-donkey-996"
    rl_steps = 10
    bo_takeover = 0.00015

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
        save_dir=(
            f"data/bo_vs_rl/simulation/hybrid_{bo_takeover}/problem_{trial_index:03d}"
        ),
    )
    env = FilterObservation(env, ["beam", "magnets", "target"])
    env = FlattenObservation(env)
    env = PolishedDonkeyCompatibility(env)
    env = NotVecNormalize(env, f"models/{model_name}/vec_normalize.pkl")
    env = RescaleAction(env, -1, 1)

    # Load models
    rl_model = TD3.load(f"models/{model_name}/model")
    bo_model = BayesianOptimizationAgent(
        env=env,
        stepsize=0.05,
        init_samples=rl_steps,
        acquisition="UCB",
        mean_module=None,
        beta=0.01,
    )

    observation = env.reset()
    done = False

    # RL agent's turn
    i = 0
    while i < rl_steps and not done:
        action, _ = rl_model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        i += 1

    if (
        bo_takeover is not None
        and unwrap_wrapper(env, RecordEpisode).infos[-1]["l1_distance"]
        > bo_takeover * 4
    ):
        print("BO is taking over")
        # Prepare env for BO
        env = unwrap_wrapper(env, RecordEpisode)
        env.unwrapped.action_mode = "direct"  # TODO direct vs direct_unidirectional?
        env.unwrapped.action_space = spaces.Box(
            low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
            high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
        )
        env.unwrapped.threshold_hold = 1
        env = RescaleAction(env, -6, 6)  # Twice the size because bidirectional

        # Retreive past examples and them feed to BO
        record_episode = unwrap_wrapper(env, RecordEpisode)

        rl_magnet_history = [
            observation_to_scaled_action(env, obs)
            for obs in record_episode.observations[1:]
        ]
        next_rl_proposal, _ = rl_model.predict(observation, deterministic=True)
        bo_model.X = torch.tensor(np.stack(rl_magnet_history + [next_rl_proposal]))

        rl_magnets = [
            observation_to_scaled_action(env, obs)
            for obs in record_episode.observations[1:]
        ]
        bo_model.X = torch.tensor(np.stack(rl_magnets))

        rl_objectives = record_episode.rewards
        bo_model.Y = torch.tensor(rl_objectives[:-1]).reshape(-1, 1)
        reward = rl_objectives[-1]

        # BO's turn
        while not done:
            action = bo_model.predict(observation, reward)
            observation, reward, done, info = env.step(action)

        # Set back to
        action = bo_model.X[bo_model.Y.argmax()].detach().numpy()
        env.step(action)

        # Take note that BO took over
        with open(
            f"data/bo_vs_rl/simulation/hybrid_{bo_takeover}/problem_{trial_index:03d}/bo_took_over.txt",
            "w",
        ) as f:
            f.write("bo_took_over")
    else:
        while not done:
            action, _ = rl_model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)

    env.close()


def main():
    trials = load_trials(Path("trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()
