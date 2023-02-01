from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gym.wrappers import RescaleAction, TimeLimit
from scipy.optimize import minimize
from tqdm.notebook import tqdm

from backend import CheetahBackend
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import RecordEpisode


def try_problem(trial_index: int, trial: Trial):
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
        w_on_screen=100.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(
        env,
        save_dir=f"data/bo_vs_rl/simulation/nelder-mead/problem_{trial_index:03d}",
    )
    RescaleAction(env, -1, 1)

    _ = env.reset()

    def objective_fun(magnets: np.ndarray) -> float:
        _, reward, _, _ = env.step(magnets)
        return -reward

    minimize(
        objective_fun,
        method="Nelder-Mead",
        x0=[0.1388888889, -0.1388888889, 0, 0.1388888889, 0],
        bounds=[(-1, 1)] * 5,
        options={
            "maxfev": 150,
            "initial_simplex": [
                [
                    0.08786718591702503,
                    -0.5128561910433027,
                    -0.1553036972951094,
                    0.6127555427773284,
                    0.2595461894779252,
                ],
                [
                    0.3079760471374653,
                    -0.07775451187952798,
                    -0.7035056011675531,
                    -0.14037664953147955,
                    -0.23283798349228,
                ],
                [
                    0.08345138641747685,
                    0.6185286060245645,
                    0.4248057538441752,
                    -0.905253799008817,
                    -0.3467166627467626,
                ],
                [
                    -0.5893341508309955,
                    0.5697893855826359,
                    -0.5030176388221985,
                    0.07204829569723259,
                    0.227008261478149,
                ],
                [
                    0.04195937605198785,
                    -0.9854189300485948,
                    -0.3045663983226312,
                    -0.5358776362398363,
                    -0.7734508884602247,
                ],
                [
                    -0.22399212545835656,
                    -0.7720730144027999,
                    0.13061285609649587,
                    0.5265235311551748,
                    0.3587917851066804,
                ],
            ],
        },
    )

    env.close()


def main():
    trials = load_trials(Path("trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(try_problem, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()
