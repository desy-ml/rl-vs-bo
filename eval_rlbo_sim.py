from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from gym import spaces
from gym.wrappers import FilterObservation, FlattenObservation, RescaleAction, TimeLimit
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import unwrap_wrapper
from tqdm.notebook import tqdm

from bayesopt import get_new_bound, get_next_samples, observation_to_scaled_action
from ea_train import ARESEACheetah
from trial import Trial, load_trials
from utils import NotVecNormalize, PolishedDonkeyCompatibility, RecordEpisode


def try_problem(trial_index: int, trial: Trial) -> None:
    model_name = "polished-donkey-996"
    rl_steps = 10
    bo_takeover = 0.00015
    stepsize = 0.05
    beta = 0.01
    acquisition = "UCB"
    mean_module = None

    # Load the model
    model = TD3.load(f"models/{model_name}/model")

    # Create the environment
    env = ARESEACheetah(
        action_mode="delta",
        incoming_mode="constant",
        incoming_values=trial.incoming_beam,
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        max_quad_delta=30 * 0.1,
        max_steerer_delta=6e-3 * 0.1,
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
        env,
        save_dir=f"data/bo_vs_rl/simulation/rl_bo_takeover_{bo_takeover}/problem_{trial_index:03d}",
    )
    env = FilterObservation(env, ["beam", "magnets", "target"])
    env = FlattenObservation(env)
    env = PolishedDonkeyCompatibility(env)
    env = NotVecNormalize(env, f"models/{model_name}/vec_normalize.pkl")
    env = RescaleAction(env, -1, 1)

    # Env needs to be setup slightly differently, so we can retreive samples for BO
    env.unwrapped.reward_mode = "feedback"
    env.unwrapped.log_beam_distance = True
    env.unwrapped.normalize_beam_distance = False

    observation = env.reset()
    done = False

    # RL agent's turn
    i = 0
    while i < rl_steps and not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        i += 1

    if (
        bo_takeover is not None
        and unwrap_wrapper(env, RecordEpisode).infos[-1]["l1_distance"]
        > bo_takeover * 4
    ):
        print("BO is taking over")
        # Prepare env for BO
        env = unwrap_wrapper(env, FlattenObservation)
        env.unwrapped.action_mode = "direct"  # TODO direct vs direct_unidirectional?
        env.unwrapped.action_space = spaces.Box(
            low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
            high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
        )
        env.unwrapped.threshold_hold = 1
        env = RescaleAction(env, -1, 1)

        # Retreive past examples and them feed to BO
        record_episode = unwrap_wrapper(env, RecordEpisode)
        # rl_magnets = [obs["magnets"] for obs in record_episode.observations]
        rl_magnets = [
            observation_to_scaled_action(env, obs)
            for obs in record_episode.observations
        ][1:]
        X = torch.tensor(rl_magnets)
        rl_objectives = record_episode.rewards
        Y = torch.tensor(rl_objectives).reshape(-1, 1)

        # BO's turn
        while not done:
            current_action = X[-1].detach().numpy()
            bounds = get_new_bound(env, current_action, stepsize)
            action_t = get_next_samples(
                X.double(),
                Y.double(),
                Y.max(),
                torch.tensor(bounds, dtype=torch.double),
                n_points=1,
                acquisition=acquisition,
                mean_module=mean_module,
                beta=beta,
            )
            action = action_t.detach().numpy().flatten()
            _, objective, done, _ = env.step(action)

            # append data
            X = torch.cat([X, action_t])
            Y = torch.cat([Y, torch.tensor([[objective]], dtype=torch.float32)])

        # Set back to
        set_to_best = True
        if set_to_best:
            action = X[Y.argmax()].detach().numpy()
            env.step(action)
    else:
        print("RL is continuing")
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)

    env.close()


def main():
    trials = load_trials(Path("trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()
