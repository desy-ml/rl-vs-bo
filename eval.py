from __future__ import annotations

import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Episode:
    """An episode of an ARES EA optimisation."""

    def __init__(self, data: dict, problem_index: Optional[int] = None):
        self.data = data
        self.problem_index = problem_index

        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def load(cls, path: Union[Path, str], use_problem_index: bool = False) -> Episode:
        """Load the data from one episode recording .pkl file."""
        if isinstance(path, str):
            path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)
        problem_index = parse_problem_index(path) if use_problem_index else None

        return cls(data, problem_index=problem_index)

    def __len__(self) -> int:
        return len(
            self.observations
        )  # Number of steps this episode ran for (including reset)

    def head(self, n: int, keep_last: bool = False) -> Episode:
        """Return an episode with only the first `n` steps of this one."""
        data_head = deepcopy(self.data)
        for key in data_head.keys():
            if key == "observations":
                data_head["observations"] = data_head["observations"][: n + 1]
                if keep_last:
                    data_head["observations"][-1] = self.data["observations"][-1]
            elif isinstance(data_head[key], list):
                data_head[key] = data_head[key][:n]
                if keep_last:
                    data_head[key][-1] = self.data[key][-1]

        return self.__class__(data_head, problem_index=self.problem_index)

    def tail(self, n: int) -> Episode:
        """Return an episode with the last `n` steps of this one."""
        data_tail = deepcopy(self.data)
        for key in data_tail.keys():
            if key == "observations":
                data_tail["observations"] = data_tail["observations"][-n - 1 :]
            elif isinstance(data_tail[key], list):
                data_tail[key] = data_tail[key][-n:]

        return self.__class__(data_tail, problem_index=self.problem_index)

    def maes(self) -> list:
        """Get the sequence of MAEs over the episdoe."""
        beams = [obs["beam"] for obs in self.observations]
        target = self.observations[0]["target"]
        maes = np.mean(np.abs(np.array(beams) - np.array(target)), axis=1).tolist()
        return maes

    def min_maes(self) -> list:
        """
        Compute the sequences of smallest MAE seen until any given step in the episode.
        """
        maes = self.maes()
        min_maes = [min(maes[: i + 1]) for i in range(len(maes))]
        return min_maes

    @property
    def target(self) -> np.ndarray:
        return self.observations[-1]["target"]

    def target_size(self) -> float:
        """Compute a measure of size for the episode's target."""
        return np.mean([self.target[1], self.target[3]])

    def steps_to_convergence(self, threshold: float = 20e-6) -> int:
        """
        Find the number of steps until the MAEs converge towards some value, i.e. change
        no more than threshold in the future.
        """
        df = pd.DataFrame({"min_mae": self.min_maes()})
        df["mae_diff"] = df["min_mae"].diff()
        df["abs_mae_diff"] = df["mae_diff"].abs()

        convergence_step = df.index.max()
        for i in df.index:
            x = all(df.loc[i:, "abs_mae_diff"] < threshold)
            if x:
                convergence_step = i
                break

        return convergence_step

    def steps_to_threshold(self, threshold: float = 20e-6) -> int:
        """
        Find the number of steps until the maes in `episdoe` drop below `threshold`.
        """
        maes = np.array(self.min_maes())
        arg_lower = np.argwhere(maes < threshold).squeeze()
        return arg_lower[0] if len(arg_lower) > 0 else len(maes)

    def plot_best_return_deviation_example(self) -> None:
        """
        Plot an example of MAE over time with markings of the location and value of the
        best setting, to help understand deviations when returning to that setting.
        """
        maes = self.maes()
        first = np.argmin(maes)

        plt.figure(figsize=(5, 3))
        plt.plot(maes)
        plt.axvline(first, c="red")
        plt.axhline(maes[first], c="green")
        plt.show()


class Study:
    """
    A study comprising multiple optimisation runs.
    """

    def __init__(self, episodes: list[Episode], name: Optional[str] = None) -> None:
        self.episodes = episodes
        self.name = name

    @classmethod
    def load(
        cls,
        data_dir: Union[Path, str],
        runs: Union[str, list[str]] = "*problem_*",
        name: Optional[str] = None,
    ) -> Study:
        """
        Loads all episode pickle files from an evaluation firectory. Expects
        `problem_xxx` directories, each of which has a `recorded_episdoe_1.pkl file in
        it.
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        run_paths = (
            data_dir.glob(runs)
            if isinstance(runs, str)
            else [data_dir / run for run in runs]
        )
        paths = [p / "recorded_episode_1.pkl" for p in run_paths]
        episodes = [Episode.load(p, use_problem_index=True) for p in paths]

        return Study(episodes, name=name)

    def __len__(self) -> int:
        return len(self.episodes)

    def head(self, n: int, keep_last: bool = False) -> Study:
        """Return study with `n` first steps from all episodes in this study."""
        return Study(
            episodes=[
                episode.head(n, keep_last=keep_last) for episode in self.episodes
            ],
            name=f"{self.name} - head",
        )

    def tail(self, n: int) -> Study:
        """Return study with `n` last steps from all episodes in this study."""
        return Study(
            episodes=[episode.tail(n) for episode in self.episodes],
            name=f"{self.name} - tail",
        )

    def problem_intersection(self, other: Study, rename: bool = False) -> Study:
        """
        Return a new study from the intersection of problems with the `other` study.
        """
        my_problems = set(self.problem_indicies())
        other_problems = set(other.problem_indicies())

        episodes = [
            self.get_episodes_by_problem(problem)[0]
            for problem in my_problems.intersection(other_problems)
        ]

        return Study(
            episodes=episodes,
            name=f"{self.name} ∩ {other.name}" if rename else self.name,
        )

    def median_best_mae(self) -> float:
        """Compute median of best MAEs seen until the very end of the episodes."""
        maes = [episode.maes() for episode in self.episodes]
        best_maes = [min(episode) for episode in maes]
        return np.median(best_maes)

    def median_final_mae(self) -> float:
        """
        Median of the final MAE that the algorithm stopped at (without returning to best
        seen).
        """
        maes = [episode.maes() for episode in self.episodes]
        final_maes = [episode[-1] for episode in maes]  # TODO Why was there index -2 ?
        return np.median(final_maes)

    def median_steps_to_convergence(self, threshold=20e-6) -> float:
        """
        Median number of steps until best seen MAE no longer improves by more than
        `threshold`.
        """
        steps = [episode.steps_to_convergence(threshold) for episode in self.episodes]
        return np.median(steps)

    def median_steps_to_threshold(self, threshold=20e-6) -> float:
        """
        Median number of steps until best seen MAE drops below (resolution) `threshold`.
        """
        steps = [episode.steps_to_threshold(threshold) for episode in self.episodes]
        return np.median(steps)

    def problem_indicies(self) -> list[int]:
        """
        Return unsorted list of problem indicies in this study. `None` is returned for
        problems that do not have a problem index.
        """
        return [episode.problem_index for episode in self.episodes]

    def get_episodes_by_problem(self, i: int) -> list[Episode]:
        """Get all episodes in this study that have problem index `i`."""
        return [episode for episode in self.episodes if episode.problem_index == i]

    def all_episodes_have_problem_index(self) -> bool:
        """
        Check if all episodes in this study have a problem index associated with them.
        """
        return all(hasattr(episode, "problem_index") for episode in self.episodes)

    def are_problems_unique(self) -> bool:
        """Check if there is at most one of each problem (index)."""
        idxs = self.problem_indicies()
        return len(idxs) == len(set(idxs))

    def plot_best_mae_over_problem(self) -> None:
        """
        Plot the best MAE achieved for each problem to see if certain problems stand
        out.
        """
        assert (
            self.all_episodes_have_problem_index()
        ), "At least on episode in this study does not have a problem index."
        assert self.are_problems_unique(), "There are duplicate problems in this study."

        sorted_problems = sorted(self.problem_indicies())
        sorted_episodes = [
            self.get_episodes_by_problem(problem)[0] for problem in sorted_problems
        ]
        final_min_maes = [episode.min_maes()[-1] for episode in sorted_episodes]

        plt.figure(figsize=(5, 3))
        plt.bar(sorted_problems, final_min_maes, label=self.name)
        plt.legend()
        plt.xlabel("Problem Index")
        plt.ylabel("Best MAE")
        plt.show()

    def plot_target_beam_size_mae_correlation(self) -> None:
        """Plot best MAEs over mean target beam size to see possible correlation."""

        best_mae = [min(episode.maes()) for episode in self.episodes]
        target_sizes = [episode.target_size() for episode in self.episodes]

        plt.figure(figsize=(5, 3))
        plt.scatter(target_sizes, best_mae, s=3, label=self.name)
        plt.legend()
        plt.xlabel("Mean beam size x/y")
        plt.ylabel("Best MAE")
        plt.show()

    def plot_best_return_deviation_box(self, save_path: str = None) -> None:
        """
        Plot a boxplot showing how far the MAE in the final return step differed from the
        MAE seen the first time the optimal magnets were set. This should show effects of
        hysteresis (and simular effects).
        """
        maes = [episode.maes() for episode in self.episodes]
        best = [min(episode) for episode in maes]
        final = [episode[-1] for episode in maes]
        deviations = np.abs(np.array(best) - np.array(final))

        plt.figure(figsize=(5, 2))
        plt.title(f"Deviation when returning to best")
        sns.boxplot(x=deviations, y=["Deviation"] * len(deviations))
        plt.grid(ls="--")
        plt.gca().set_axisbelow(True)
        plt.xlabel("MAE")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()


def problem_aligned(studies: list[Study]) -> list[Study]:
    """
    Intersect the problems of all `studies` such that the studies in the returned list
    all cover exactly the same problems.
    """
    # Find the smallest intersection of problem indicies
    intersected = set(studies[0].problem_indicies())
    for study in studies:
        intersected = intersected.intersection(set(study.problem_indicies()))

    new_studies = []
    for study in studies:
        intersected_study = Study(
            episodes=[study.get_episodes_by_problem(i)[0] for i in intersected],
            name=study.name,
        )
        new_studies.append(intersected_study)

    return new_studies


def parse_problem_index(path: Path) -> int:
    """
    Take a `path` to an episode recording according to a problems file and parse the
    problem index for it. Assumes that the recording is in some subdirectory of shape
    `*problem_*`.
    """
    return int(path.parent.name.split("_")[-1])


def plot_best_mae_box(studies: list[Study], save_path: str = None) -> None:
    """Box plot of best MAEs seen until the very end of the episodes."""
    combined_best_maes = []
    combined_names = []
    for study in studies:
        maes = [episode.maes() for episode in study.episodes]
        best_maes = [min(episode) for episode in maes]

        names = [study.name] * len(best_maes)

        combined_best_maes += best_maes
        combined_names += names

    plt.figure(figsize=(5, 0.6 * len(studies)))
    plt.title("Best MAEs")
    sns.boxplot(x=combined_best_maes, y=combined_names)
    plt.xscale("log")
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_best_mae_diff_over_problem(
    study_1: Study,
    study_2: Study,
    save_path: str = None,
) -> None:
    """Plot the differences of the best MAE achieved for each problem to see if certain problems stand out."""
    assert study_1.are_problems_unique(), "The problems in study 1 are note unique."
    assert study_2.are_problems_unique(), "The problems in study 2 are note unique."

    study_1_idxs = sorted(study_1.problem_indicies())
    study_2_idxs = sorted(study_2.problem_indicies())
    assert study_1_idxs == study_2_idxs, "The studies do not cover the same problems."

    problem_idxs = study_1_idxs
    best_maes_1 = [
        min(study_1.get_episodes_by_problem(i)[0].maes()) for i in problem_idxs
    ]
    best_maes_2 = [
        min(study_2.get_episodes_by_problem(i)[0].maes()) for i in problem_idxs
    ]

    diff = np.array(best_maes_1) - np.array(best_maes_2)

    plt.figure(figsize=(5, 3))
    plt.bar(problem_idxs, diff, label=f"{study_1.name} vs. {study_2.name}")
    plt.legend()
    plt.xlabel("Problem Index")
    plt.ylabel("Best MAE")
    plt.grid()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_best_mae_over_time(
    studies: list[Study], threshold: Optional[float] = None, save_path: str = None
) -> None:
    """
    Plot mean best seen MAE over all episdoes over time. Optionally display a
    `threshold` line to mark measurement limit.
    """
    dfs = []
    for study in studies:
        ds = [
            {
                "min_mae": episode.min_maes(),
                "step": range(len(episode)),
                "problem": episode.problem_index,
                "study_name": study.name,
            }
            for episode in study.episodes
        ]
        df = pd.concat(pd.DataFrame(d) for d in ds)

        dfs.append(df)

    combined_df = pd.concat(dfs)

    plt.figure(figsize=(5, 3))
    if threshold is not None:
        plt.axhline(threshold, ls="--", color="lightsteelblue", label="Threshold")
    sns.lineplot(x="step", y="min_mae", hue="study_name", data=combined_df)
    plt.title("Mean Best MAE Over Time")
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_final_mae_box(studies: list[Study], save_path: str = None) -> None:
    """
    Box plot of the final MAE that the algorithm stopped at (without returning to best
    seen).
    """
    combined_final_maes = []
    combined_names = []
    for study in studies:
        maes = [episode.maes() for episode in study.episodes]
        final_maes = [episode[-1] for episode in maes]  # TODO Used to be index -2 ?

        names = [study.name] * len(final_maes)

        combined_final_maes += final_maes
        combined_names += names

    plt.figure(figsize=(5, 0.6 * len(studies)))
    plt.title("Final MAEs")
    sns.boxplot(x=combined_final_maes, y=combined_names)
    plt.xscale("log")
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_mae_over_time(
    studies: list[Study], save_path: str = None, threshold: Optional[float] = None
) -> None:
    """
    Plot mean MAE of over episodes over time. Optionally display a `threshold` line to
    mark measurement limit.
    """
    dfs = []
    for study in studies:
        ds = [
            {
                "mae": episode.maes(),
                "step": range(len(episode)),
                "problem": episode.problem_index,
                "study_name": study.name,
            }
            for episode in study.episodes
        ]
        df = pd.concat(pd.DataFrame(d) for d in ds)

        dfs.append(df)

    combined_df = pd.concat(dfs)

    plt.figure(figsize=(5, 3))
    if threshold is not None:
        plt.axhline(threshold, ls="--", color="lightsteelblue", label="Threshold")
    sns.lineplot(x="step", y="mae", hue="study_name", data=combined_df)
    plt.title("Mean MAE Over Time")
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_steps_to_convergence_box(
    studies: list[Study],
    threshold: float = 20e-6,
    save_path: Optional[str] = None,
) -> None:
    """
    Box plot number of steps until best seen MAE no longer improves by more than
    `threshold`.
    """
    combined_steps = []
    combined_names = []
    for study in studies:
        steps = [episode.steps_to_convergence(threshold) for episode in study.episodes]
        names = [study.name] * len(steps)

        combined_steps += steps
        combined_names += names

    plt.figure(figsize=(5, 0.6 * len(studies)))
    plt.title(f"Steps to convergence (limit = {threshold})")
    sns.boxplot(x=combined_steps, y=combined_names)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.xlabel("No. of steps")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_steps_to_threshold_box(
    studies: list[Study],
    threshold: float = 20e-6,
    save_path: Optional[str] = None,
) -> None:
    """
    Box plot number of steps until best seen MAE drops below (resolution) `threshold`.
    """
    combined_steps = []
    combined_names = []
    for study in studies:
        steps = [episode.steps_to_threshold(threshold) for episode in study.episodes]
        names = [study.name] * len(steps)

        combined_steps += steps
        combined_names += names

    plt.figure(figsize=(5, 0.6 * len(studies)))
    plt.title(f"Steps to MAE below {threshold}")
    sns.boxplot(x=combined_steps, y=combined_names)
    plt.grid(ls="--")
    plt.gca().set_axisbelow(True)
    plt.xlabel("No. of steps")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
