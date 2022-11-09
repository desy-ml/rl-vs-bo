import os
import pickle
from glob import glob
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def compute_final_min_mae(episode: list) -> float:
    """Compute list of the smallest MAE seen the run over the course of the entire episode."""
    maes = get_maes(episode)
    min_maes = compute_min_maes(maes)

    return min_maes[-1]


def compute_min_maes(maes: list) -> list:
    """From the sequence of MAEs compute the sequence of lowest already seen MAEs."""
    min_maes = [min(maes[: i + 1]) for i in range(len(maes))]
    return min_maes


def compute_target_size(episode: list) -> float:
    """Compute a measure for the size of the target beam."""
    target = episode["observations"][-1]["target"]
    # return np.min([target[1], target[3]])
    # return target[1] * target[3]
    return np.mean([target[1], target[3]])


def find_convergence(episode: list, threshold: float = 20e-6) -> int:
    """
    Find the number of steps until the MAEs converge towards some value, i.e. change no
    more than threshold in the future.
    """

    df = pd.DataFrame({"min_mae": episode})
    df["mae_diff"] = df["min_mae"].diff()
    df["abs_mae_diff"] = df["mae_diff"].abs()

    convergence_step = df.index.max()
    for i in df.index:
        x = all(df.loc[i:, "abs_mae_diff"] < threshold)
        if x:
            convergence_step = i
            break

    return convergence_step


def full_evaluation(rl: list[dict], bo: list[dict], save_dir: str = None) -> None:
    """
    Fully evaluate a number of things about two different algorithms, showing plots and
    metrics.
    """
    if save_dir is not None:
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    print(f"Evaluating rl = {len(rl)} vs. bo = {len(bo)} problems")

    print_seperator()

    print(f"RL -> {median_steps_to_threshold(rl)}")
    print(f"BO -> {median_steps_to_threshold(bo)}")
    plot_steps_to_threshold_box(
        {"RL": rl, "BO": bo},
        save_path=f"{save_dir}/steps_to_target.pdf" if save_dir is not None else None,
    )

    print_seperator()

    print(f"RL -> {median_steps_to_convergence(rl)}")
    print(f"BO -> {median_steps_to_convergence(bo)}")

    plot_steps_to_convergence_box(
        {"RL": rl, "BO": bo},
        save_path=f"{save_dir}/steps_to_convergence.pdf"
        if save_dir is not None
        else None,
    )

    print_seperator()

    plot_mae_over_time(
        {"RL": rl, "BO": bo},
        threshold=20e-6,
        save_path=f"{save_dir}/mae_over_time.pdf" if save_dir is not None else None,
    )

    print_seperator()

    plot_best_mae_over_time(
        {"RL": rl, "BO": bo},
        threshold=20e-6,
        save_path=f"{save_dir}/best_mae_over_time.pdf"
        if save_dir is not None
        else None,
    )

    print_seperator()

    print(f"RL -> {median_final_mae(rl)}")
    print(f"BO -> {median_final_mae(bo)}")

    plot_final_mae_box(
        {"RL": rl, "BO": bo},
        save_path=f"{save_dir}/final_mae.pdf" if save_dir is not None else None,
    )

    print_seperator()

    print(f"RL -> {median_best_mae(rl)}")
    print(f"BO -> {median_best_mae(bo)}")

    plot_best_mae_box(
        {"RL": rl, "BO": bo},
        save_path=f"{save_dir}/final_best_mae.pdf" if save_dir is not None else None,
    )


def get_maes(episode: dict) -> list:
    """Get sequence of step-wise MAEs from episode data in `episode`."""
    beams = [obs["beam"] for obs in episode["observations"]]
    target = episode["observations"][0]["target"]
    maes = np.mean(np.abs(np.array(beams) - np.array(target)), axis=1).tolist()

    return maes


def get_steps_to_treshold(episode: list, threshold: float = 20e-6) -> int:
    """Find the number of steps until the maes in `episdoe` drop below `threshold`."""
    episode = np.array(episode)
    arg_lower = np.argwhere(episode < threshold).squeeze()
    return arg_lower[0] if len(arg_lower) > 0 else len(episode)


def load_episode_data(path: str) -> dict:
    """Load the data from one episode recording .pkl file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_eval_data(eval_dir: str, progress_bar: bool = False) -> dict[dict]:
    """
    Load all episode pickle files from an evaluation firectory. Expects `problem_xxx`
    directories, each of which has a `recorded_episdoe_1.pkl file in it.
    """
    paths = sorted(glob(f"{eval_dir}/*problem_*/recorded_episode_1.pkl"))
    if progress_bar:
        paths = tqdm(paths)
    data = [load_episode_data(p) for p in paths]
    idxs = [parse_problem_index(p) for p in paths]
    data_dict = {i: rec for i, rec in zip(idxs, data)}

    return data_dict


def median_best_mae(data: list[dict]) -> float:
    """Compute median of best MAEs seen until the very end of the episodes."""
    maes = [get_maes(episode) for episode in data]
    final_maes = [min(episode) for episode in maes]
    return np.median(final_maes)


def median_final_mae(data: list[dict]) -> float:
    """
    Median of the final MAE that the algorithm stopped at (without returning to best
    seen).
    """
    maes = [get_maes(episode) for episode in data]
    final_maes = [episode[-2] for episode in maes]
    return np.median(final_maes)


def median_steps_to_convergence(data: list[dict], threshold=20e-6) -> float:
    """
    Median number of steps until best seen MAE no longer improves by more than
    `threshold`.
    """
    maes = [get_maes(episode) for episode in data]
    min_maes = [compute_min_maes(episode) for episode in maes]
    steps = [find_convergence(episode, threshold) for episode in min_maes]
    return np.median(steps)


def median_steps_to_threshold(data: dict, threshold=20e-6) -> float:
    """
    Median number of steps until best seen MAE drops below (resolution) `threshold`.
    """
    maes = [get_maes(episode) for episode in data]
    min_maes = [compute_min_maes(episode) for episode in maes]
    steps = [get_steps_to_treshold(episode, threshold) for episode in min_maes]
    return np.median(steps)


def parse_problem_index(path: str) -> int:
    """
    Take a `path` to an episode recording according to a problems file and parse the
    problem index for it. Assumes that the recording is in some subdirectory of shape
    `*problem_*`.
    """
    directories = path.split("/")
    for d in directories:
        if "problem" in d:
            problem_string = d

    return int(problem_string.split("problem_")[-1])


def plot_best_mae_box(data: dict[dict], save_path: str = None) -> None:
    """Box plot of best MAEs seen until the very end of the episodes."""
    combined_final_maes = []
    combined_methods = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results.values()]
        final_maes = [min(episode) for episode in maes]

        methods = [method] * len(final_maes)

        combined_final_maes += final_maes
        combined_methods += methods

    plt.figure(figsize=(5, 0.6 * len(data)))
    plt.title("Best MAEs")
    sns.boxplot(x=combined_final_maes, y=combined_methods)
    plt.xscale("log")
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_best_mae_diff_over_problem(
    results_1: dict[dict],
    results_2: dict[dict],
    name_1: str = None,
    name_2: str = None,
    save_path: str = None,
) -> None:
    """Plot the differences of the best MAE achieved for each problem to see if certain problems stand out."""
    assert set(results_1.keys()) == set(
        results_2.keys()
    ), "Results 1 and 2 do not cover the same set of problems."

    problem_idxs = results_1.keys()
    final_min_maes_1 = [compute_final_min_mae(results_1[p]) for p in problem_idxs]
    final_min_maes_2 = [compute_final_min_mae(results_2[p]) for p in problem_idxs]

    diff = np.array(final_min_maes_1) - np.array(final_min_maes_2)

    plt.figure(figsize=(5, 3))
    plt.bar(problem_idxs, diff, label=f"{name_1} vs. {name_2}")
    plt.legend()
    plt.xlabel("Problem Index")
    plt.ylabel("Best MAE")
    plt.grid()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_best_mae_over_problem(results: dict[dict], name: str = None) -> None:
    """Plot the best MAE achieved for each problem to see if certain problems stand out."""
    problem_idxs = results.keys()
    final_min_maes = [compute_final_min_mae(results[p]) for p in problem_idxs]

    plt.figure(figsize=(5, 3))
    plt.bar(problem_idxs, final_min_maes, label=name)
    plt.legend()
    plt.xlabel("Problem Index")
    plt.ylabel("Best MAE")
    plt.show()


def plot_best_mae_over_time(
    data: dict[dict], threshold: Optional[float] = None, save_path: str = None
) -> None:
    """
    Plot mean best seen MAE over all episdoes over time. Optionally display a
    `threshold` line to mark measurement limit.
    """
    dfs = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results.values()]
        min_maes = [compute_min_maes(episode) for episode in maes]

        ds = [
            {
                "mae": episode,
                "step": range(len(episode)),
                "problem": i,
                "method": method,
            }
            for i, episode in enumerate(min_maes)
        ]
        df = pd.concat(pd.DataFrame(d) for d in ds)

        dfs.append(df)

    combined_df = pd.concat(dfs)

    plt.figure(figsize=(5, 3))
    if threshold is not None:
        plt.axhline(threshold, ls="--", color="lightsteelblue", label="Threshold")
    sns.lineplot(x="step", y="mae", hue="method", data=combined_df)
    plt.title("Mean Best MAE Over Time")
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_final_mae_box(data: dict[dict], save_path: str = None) -> None:
    """
    Box plot of the final MAE that the algorithm stopped at (without returning to best
    seen).
    """
    combined_final_maes = []
    combined_methods = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results.values()]
        final_maes = [episode[-2] for episode in maes]

        methods = [method] * len(final_maes)

        combined_final_maes += final_maes
        combined_methods += methods

    plt.figure(figsize=(5, 0.6 * len(data)))
    plt.title("Final MAEs")
    sns.boxplot(x=combined_final_maes, y=combined_methods)
    plt.xscale("log")
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_mae_over_time(
    data: dict[dict], save_path: str = None, threshold: Optional[float] = None
) -> None:
    """
    Plot mean MAE of over episodes over time. Optionally display a `threshold` line to
    mark measurement limit.
    """
    dfs = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results.values()]

        ds = [
            {
                "mae": episode,
                "step": range(len(episode)),
                "problem": i,
                "method": method,
            }
            for i, episode in enumerate(maes)
        ]
        df = pd.concat(pd.DataFrame(d) for d in ds)

        dfs.append(df)

    combined_df = pd.concat(dfs)

    plt.figure(figsize=(5, 3))
    if threshold is not None:
        plt.axhline(threshold, ls="--", color="lightsteelblue", label="Threshold")
    sns.lineplot(x="step", y="mae", hue="method", data=combined_df)
    plt.title("Mean MAE Over Time")
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_steps_to_convergence_box(
    data: dict[dict], threshold=20e-6, save_path: str = None
) -> None:
    """
    Box plot number of steps until best seen MAE no longer improves by more than
    `threshold`.
    """
    combined_steps = []
    combined_methods = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results.values()]
        min_maes = [compute_min_maes(episode) for episode in maes]
        steps = [find_convergence(episode, threshold) for episode in min_maes]

        methods = [method] * len(steps)

        combined_steps += steps
        combined_methods += methods

    plt.figure(figsize=(5, 0.6 * len(data)))
    plt.title(f"Steps to convergence (limit = {threshold})")
    sns.boxplot(x=combined_steps, y=combined_methods)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.xlabel("No. of steps")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_steps_to_threshold_box(
    data: dict[dict],
    threshold=20e-6,
    save_path: str = None,
) -> None:
    """
    Box plot number of steps until best seen MAE drops below (resolution) `threshold`.
    """
    combined_steps = []
    combined_methods = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results.values()]
        min_maes = [compute_min_maes(episode) for episode in maes]
        steps = [get_steps_to_treshold(episode, threshold) for episode in min_maes]

        methods = [method] * len(steps)

        combined_steps += steps
        combined_methods += methods

    plt.figure(figsize=(5, 0.6 * len(data)))
    plt.title(f"Steps to MAE below {threshold}")
    sns.boxplot(x=combined_steps, y=combined_methods)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.xlabel("No. of steps")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_target_beam_size_mae_correlation(result: dict[dict], name: str = None) -> None:
    """Plot best MAEs over mean target beam size to see possible correlation."""

    final_min_maes = [compute_final_min_mae(episode) for episode in result.values()]
    target_sizes = [compute_target_size(episode) for episode in result.values()]

    plt.figure(figsize=(5, 3))
    plt.scatter(target_sizes, final_min_maes, s=3, label=name)
    plt.legend()
    plt.xlabel("Mean beam size x/y")
    plt.ylabel("Best MAE")
    plt.show()


def print_seperator() -> None:
    """Print a seperator line to help structure outputs."""
    print("-----------------------------------------------------------")
