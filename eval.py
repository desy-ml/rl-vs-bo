import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def compute_min_maes(maes: list) -> list:
    """From the sequence of MAEs compute the sequence of lowest already seen MAEs."""
    min_maes = [min(maes[: i + 1]) for i in range(len(maes))]
    return min_maes


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


def load_eval_data(eval_dir: str, progress_bar: bool = False) -> list[dict]:
    """
    Load all episode pickle files from an evaluation firectory. Expects `problem_xxx`
    directories, each of which has a `recorded_episdoe_1.pkl file in it.
    """
    paths = sorted(glob(f"{eval_dir}/problem_*/recorded_episode_1.pkl"))
    if progress_bar:
        paths = tqdm(paths)
    data = [load_episode_data(p) for p in paths]
    return data


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


def plot_best_mae_box(data: dict) -> None:
    """Box plot of best MAEs seen until the very end of the episodes."""
    combined_final_maes = []
    combined_methods = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results]
        final_maes = [min(episode) for episode in maes]

        methods = [method] * len(final_maes)

        combined_final_maes += final_maes
        combined_methods += methods

    plt.figure(figsize=(5, 2))
    plt.title("Best MAEs")
    sns.boxplot(x=combined_final_maes, y=combined_methods)
    plt.xscale("log")
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def plot_best_mae_over_time(data: dict) -> None:
    """Plot mean best seen MAE over all episdoes over time."""
    dfs = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results]
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
    sns.lineplot(x="step", y="mae", hue="method", data=combined_df)
    plt.title("Mean Best MAE Over Time")
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.show()


def plot_final_mae_box(data: dict) -> None:
    """
    Box plot of the final MAE that the algorithm stopped at (without returning to best
    seen).
    """
    combined_final_maes = []
    combined_methods = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results]
        final_maes = [episode[-2] for episode in maes]

        methods = [method] * len(final_maes)

        combined_final_maes += final_maes
        combined_methods += methods

    plt.figure(figsize=(5, 2))
    plt.title("Final MAEs")
    sns.boxplot(x=combined_final_maes, y=combined_methods)
    plt.xscale("log")
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def plot_mae_over_time(data: dict) -> None:
    """Plot mean MAE of over episodes over time."""
    dfs = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results]

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
    sns.lineplot(x="step", y="mae", hue="method", data=combined_df)
    plt.title("Mean MAE Over Time")
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.show()


def plot_steps_to_convergence_box(data: dict, threshold=20e-6) -> None:
    """
    Box plot number of steps until best seen MAE no longer improves by more than
    `threshold`.
    """
    combined_steps = []
    combined_methods = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results]
        min_maes = [compute_min_maes(episode) for episode in maes]
        steps = [find_convergence(episode, threshold) for episode in min_maes]

        methods = [method] * len(steps)

        combined_steps += steps
        combined_methods += methods

    plt.figure(figsize=(5, 2))
    plt.title(f"Steps to convergence (limit = {threshold})")
    sns.boxplot(x=combined_steps, y=combined_methods)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.xlabel("No. of steps")
    plt.tight_layout()
    plt.show()


def plot_steps_to_threshold_box(data: dict, threshold=20e-6) -> None:
    """
    Box plot number of steps until best seen MAE drops below (resolution) `threshold`.
    """
    combined_steps = []
    combined_methods = []
    for method, results in data.items():
        maes = [get_maes(episode) for episode in results]
        min_maes = [compute_min_maes(episode) for episode in maes]
        steps = [get_steps_to_treshold(episode, threshold) for episode in min_maes]

        methods = [method] * len(steps)

        combined_steps += steps
        combined_methods += methods

    plt.figure(figsize=(5, 2))
    plt.title(f"Steps to MAE below {threshold}")
    sns.boxplot(x=combined_steps, y=combined_methods)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.xlabel("No. of steps")
    plt.tight_layout()
    plt.show()
