from __future__ import annotations

import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.ndimage import uniform_filter


class Episode:
    """An episode of an ARES EA optimisation."""

    def __init__(
        self,
        observations: list[Union[dict, np.ndarray]],
        rewards: list[float],
        infos: list[dict],
        actions: list[np.ndarray],
        t_start: datetime,
        t_end: datetime,
        steps_taken: int,
        step_start_times: Optional[
            list[datetime]
        ] = None,  # Optional because not all recordings have them
        step_end_times: Optional[
            list[datetime]
        ] = None,  # Optional because not all recordings have them
        problem_index: Optional[int] = None,
    ):
        self.observations = observations
        self.rewards = rewards
        self.infos = infos
        self.actions = actions
        self.t_start = t_start
        self.t_end = t_end
        self.steps_taken = steps_taken
        self.step_start_times = step_start_times
        self.step_end_times = step_end_times
        self.problem_index = problem_index

    @classmethod
    def load(
        cls,
        path: Union[Path, str],
        use_problem_index: bool = False,
        drop_screen_images: bool = False,
    ) -> Episode:
        """Load the data from one episode recording .pkl file."""
        if isinstance(path, str):
            path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)
        problem_index = parse_problem_index(path) if use_problem_index else None

        loaded = cls(**data, problem_index=problem_index)

        if drop_screen_images:
            loaded.drop_screen_images()

        return loaded

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

    def final_mae(self) -> float:
        """MAE at the end of the episode."""
        return self.maes()[-1]

    def best_mae(self) -> float:
        """Best MAE observed over the entire duration of the episode."""
        return min(self.maes())

    def rmse(self) -> float:
        """
        RMSE over all samples in episode and over all beam parameters as used in
        https://www.nature.com/articles/s41586-021-04301-9.
        """
        beams = np.stack([obs["beam"] for obs in self.observations])
        targets = np.stack([obs["target"] for obs in self.observations])
        rmse = np.sqrt(np.mean(np.square(targets - beams)))
        return rmse

    def abs_delta_beam_parameters(self) -> np.ndarray:
        """Get the sequence of mu_x over the episdoe."""
        beams = [obs["beam"] for obs in self.observations]
        target = self.observations[0]["target"]
        abs_deltas = np.abs(np.array(beams) - np.array(target))
        return abs_deltas

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

    def plot_beam_parameters(
        self,
        show_target: bool = True,
        vertical_marker: Union[float, tuple[float, str]] = None,
        title: Optional[str] = None,
        xlabel: bool = True,
        ylabel: bool = True,
        figsize: tuple[float, float] = (6, 3),
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[str] = None,
    ) -> matplotlib.axes.Axes:
        """
        Plot beam parameters over the episode and optionally add the target beam
        parameters if `show_target` is `True`. A vertical line to mark a point in time
        may be added via `vertical_marker` either by just its position as a float or by
        a tuple of its position and the string label that should be shown in the legend.
        """
        beams = [obs["beam"] for obs in self.observations]
        targets = [obs["target"] for obs in self.observations]

        palette_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if isinstance(vertical_marker, (int, float)):
            ax.axvline(vertical_marker, ls="--", color=palette_colors[5])
        elif isinstance(vertical_marker, tuple):
            marker_position, marker_label = vertical_marker
            ax.axvline(
                marker_position, label=marker_label, ls="--", color=palette_colors[5]
            )

        ax.plot(np.array(beams)[:, 0] * 1e6, label=r"$\mu_x$", c=palette_colors[0])
        ax.plot(np.array(beams)[:, 1] * 1e6, label=r"$\sigma_x$", c=palette_colors[1])
        ax.plot(np.array(beams)[:, 2] * 1e6, label=r"$\mu_y$", c=palette_colors[2])
        ax.plot(np.array(beams)[:, 3] * 1e6, label=r"$\sigma_y$", c=palette_colors[3])

        if show_target:
            ax.plot(np.array(targets)[:, 0] * 1e6, c=palette_colors[0], ls="--")
            ax.plot(np.array(targets)[:, 1] * 1e6, c=palette_colors[1], ls="--")
            ax.plot(np.array(targets)[:, 2] * 1e6, c=palette_colors[2], ls="--")
            ax.plot(np.array(targets)[:, 3] * 1e6, c=palette_colors[3], ls="--")

        ax.set_title(title)
        ax.legend()

        if xlabel:
            ax.set_xlabel("Step")
        if ylabel:
            ax.set_ylabel(r"Beam Parameter ($\mu$m)")

        if save_path is not None:
            assert fig is not None, "Cannot save figure when axes was passed."
            fig.savefig(save_path)

        return ax

    def plot_magnets(
        self,
        xlabel: bool = True,
        ylabel_left: bool = True,
        ylabel_right: bool = True,
        figsize: tuple[float, float] = (6, 3),
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot magnet values over episdoe."""
        magnets = np.array([obs["magnets"] for obs in self.observations])

        palette_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(magnets[:, 0], c=palette_colors[0], label="Q1")
        ax.plot(magnets[:, 1], c=palette_colors[1], label="Q2")
        ax.plot([], c=palette_colors[2], label="CV")  # Dummy for legend
        ax.plot(magnets[:, 3], c=palette_colors[3], label="Q3")
        ax.plot([], c=palette_colors[4], label="CH")  # Dummy for legend
        ax.set_xlim(0, None)
        ax.set_ylim(-72, 72)
        ax.legend()

        ax_twinx = ax.twinx()

        ax_twinx.plot(magnets[:, 2] * 1e3, c=palette_colors[2], label="CV")
        ax_twinx.plot(magnets[:, 4] * 1e3, c=palette_colors[4], label="CH")
        ax_twinx.set_ylabel("Steering Angle (mrad)")
        ax_twinx.set_ylim(-6.1782, 6.1782)

        if xlabel:
            ax.set_xlabel("Step")
        if ylabel_left:
            ax.set_ylabel(r"Quadrupole Strength ($m^{-2}$)")
        if ylabel_right:
            ax_twinx.set_ylabel("Steering Angle (mrad)")

        if save_path is not None:
            assert fig is not None, "Cannot save figure when axes was passed."
            fig.savefig(save_path)

        return ax

    def plot_maes(self, show_best_mae: bool = True) -> None:
        """
        Plot MAE over time. If `show_best_mae` is `True`, add best MAE seen up to a
        certain point to the plot.
        """
        plt.figure(figsize=(6, 3))
        plt.plot(self.maes(), label="MAE")
        if show_best_mae:
            plt.plot(self.min_maes(), label="Best MAE")
            plt.legend()
        plt.ylabel("MAE (m)")
        plt.xlabel("Step")

        plt.show()

    def drop_screen_images(self) -> None:
        """
        Drop all screen images from this loaded copy of the episode. This can help to
        save RAM while working with the data, when the images are not needed.
        """
        for info in self.infos:
            info.pop("beam_image", None)
            info.pop("screen_before_reset", None)
            info.pop("screen_after_reset", None)

    def plot_summary(
        self,
        title: Optional[str] = None,
        figsize: tuple[float, float] = (7.0, 3.5),
        save_path: Optional[str] = None,
    ) -> None:
        """Summary plot of important data about this episode."""
        fig, axs = plt.subplots(2, 2, figsize=figsize, sharex="col")

        fig.suptitle(title)

        self.plot_beam_parameters(ax=axs[0, 0], xlabel=False)

        self.plot_magnets(ax=axs[1, 0])

        # Plot screen images if there are any
        if "beam_image" in self.infos[0]:
            first_screen_image = (
                self.infos[0]["screen_before_reset"]
                if "screen_before_reset" in self.infos[0]
                else self.infos[0]["beam_image"]
            )
            last_screen_image = self.infos[-1]["beam_image"]
            shared_vmax = np.max(
                np.stack(
                    [
                        uniform_filter(first_screen_image, size=10),
                        uniform_filter(last_screen_image, size=10),
                    ]
                )
            )
            plot_screen_image(
                image=first_screen_image,
                resolution=self.infos[0]["screen_resolution"],
                pixel_size=self.infos[0]["pixel_size"],
                ax=axs[0, 1],
                vmax=shared_vmax,
                xlabel=False,
            )
            plot_screen_image(
                image=last_screen_image,
                resolution=self.infos[-1]["screen_resolution"],
                pixel_size=self.infos[-1]["pixel_size"],
                ax=axs[1, 1],
                vmax=shared_vmax,
            )

        # Plot target beam parameters
        initial_beam = (
            self.infos[0]["beam_before_reset"]
            if "beam_before_reset" in self.infos[0]
            else self.observations[0]["beam"]
        )
        target_beam = self.observations[0]["target"]
        plot_beam_parameters_on_screen(
            mu_x=initial_beam[0],
            sigma_x=initial_beam[1],
            mu_y=initial_beam[2],
            sigma_y=initial_beam[3],
            resolution=self.infos[0]["screen_resolution"],
            pixel_size=self.infos[0]["pixel_size"],
            ax=axs[0, 1],
            xlabel=False,
        )
        plot_beam_parameters_on_screen(
            mu_x=target_beam[0],
            sigma_x=target_beam[1],
            mu_y=target_beam[2],
            sigma_y=target_beam[3],
            resolution=self.infos[0]["screen_resolution"],
            pixel_size=self.infos[0]["pixel_size"],
            ax=axs[1, 1],
        )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()


class Study:
    """
    A study comprising multiple optimisation runs.
    """

    def __init__(self, episodes: list[Episode], name: Optional[str] = None) -> None:
        assert len(episodes) > 0, "No episodes passed to study at initialisation."

        self.episodes = episodes
        self.name = name

    @classmethod
    def load(
        cls,
        data_dir: Union[Path, str],
        runs: Union[str, list[str]] = "*problem_*",
        name: Optional[str] = None,
        drop_screen_images: bool = False,
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
        paths = [p / "recorded_episode_1.pkl" for p in sorted(run_paths)]
        episodes = [
            Episode.load(
                p, use_problem_index=True, drop_screen_images=drop_screen_images
            )
            for p in paths
        ]

        return Study(episodes, name=name)

    def __len__(self) -> int:
        """A study's length is the number of episodes in it."""
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
        return np.median([episode.best_mae() for episode in self.episodes])

    def median_final_mae(self) -> float:
        """
        Median of the final MAE that the algorithm stopped at (without returning to best
        seen).
        """
        return np.median([episode.best_mae() for episode in self.episodes])

    def mean_final_mae(self) -> float:
        """
        Mean of the final MAE that the algorithm stopped at (without returning to best
        seen).
        """
        return np.mean([episode.best_mae() for episode in self.episodes])

    def median_steps_to_convergence(
        self, threshold: float = 20e-6, max_steps: Optional[int] = None
    ) -> Optional[float]:
        """
        Median number of steps until best seen MAE no longer improves by more than
        `threshold`. If `max_steps` is given, only consider episodes that converged in
        less than `max_steps`. Returns `None` if no runs got there in less than
        `max_steps`.
        """
        steps = [episode.steps_to_convergence(threshold) for episode in self.episodes]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.median(steps) if len(steps) > 0 else None

    def mean_steps_to_convergence(
        self, threshold: float = 20e-6, max_steps: Optional[int] = None
    ) -> Optional[float]:
        """
        Mean number of steps until best seen MAE no longer improves by more than
        `threshold`. If `max_steps` is given, only consider episodes that converged in
        less than `max_steps`. Returns `None` if no runs got there in less than
        `max_steps`.
        """
        steps = [episode.steps_to_convergence(threshold) for episode in self.episodes]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.mean(steps) if len(steps) > 0 else None

    def median_steps_to_threshold(
        self, threshold: float = 20e-6, max_steps: Optional[int] = None
    ) -> Optional[float]:
        """
        Median number of steps until best seen MAE drops below (resolution) `threshold`.
        If `max_steps` is given, only consider episodes that got below threshold in less
        than `max_steps`. Returns `None` if no runs got there in less than `max_steps`.
        """
        steps = [episode.steps_to_threshold(threshold) for episode in self.episodes]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.median(steps) if len(steps) > 0 else None

    def mean_steps_to_threshold(
        self, threshold: float = 20e-6, max_steps: Optional[int] = None
    ) -> Optional[float]:
        """
        Mean number of steps until best seen MAE drops below (resolution) `threshold`.
        If `max_steps` is given, only consider episodes that got below threshold in less
        than `max_steps`. Returns `None` if no runs got there in less than `max_steps`.
        """
        steps = [episode.steps_to_threshold(threshold) for episode in self.episodes]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.mean(steps) if len(steps) > 0 else None

    def rmse(self, max_steps: Optional[int] = None) -> float:
        """
        RMSE over all samples in all episode in the study and over all beam parameters
        as used in https://www.nature.com/articles/s41586-021-04301-9.
        """
        beams = np.stack(
            [
                obs["beam"]
                for episode in self.episodes
                for obs in episode.observations[:max_steps]
            ]
        )
        targets = np.stack(
            [
                obs["target"]
                for episode in self.episodes
                for obs in episode.observations[:max_steps]
            ]
        )
        rmse = np.sqrt(np.mean(np.square(targets - beams)))
        return rmse

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
        best_maes = [episode.best_mae() for episode in sorted_episodes]

        plt.figure(figsize=(5, 3))
        plt.bar(sorted_problems, best_maes, label=self.name)
        plt.legend()
        plt.xlabel("Problem Index")
        plt.ylabel("Best MAE")
        plt.show()

    def plot_target_beam_size_mae_correlation(self) -> None:
        """Plot best MAEs over mean target beam size to see possible correlation."""

        best_mae = [episode.best_mae() for episode in self.episodes]
        target_sizes = [episode.target_size() for episode in self.episodes]

        plt.figure(figsize=(5, 3))
        plt.scatter(target_sizes, best_mae, s=3, label=self.name)
        plt.legend()
        plt.xlabel("Mean beam size x/y")
        plt.ylabel("Best MAE")
        plt.show()

    def plot_best_return_deviation_box(
        self, print_results: bool = True, save_path: Optional[str] = None
    ) -> None:
        """
        Plot a boxplot showing how far the MAE in the final return step differed from
        the MAE seen the first time the optimal magnets were set. This should show
        effects of hysteresis (and simular effects).
        """
        best = [episode.best_mae() for episode in self.episodes]
        final = [episode.final_mae() for episode in self.episodes]
        deviations = np.abs(np.array(best) - np.array(final))

        if print_results:
            print(f"Median deviation = {np.median(deviations)}")
            print(f"Max deviation = {np.max(deviations)}")

        plt.figure(figsize=(5, 2))
        plt.title("Deviation when returning to best")
        sns.boxplot(x=deviations, y=["Deviation"] * len(deviations))
        plt.gca().set_axisbelow(True)
        plt.xlabel("MAE")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def average_inference_times(self):
        """
        Return average time it took to infer the next action/magnet setting throughout
        the study.
        """
        first_inferences = [
            episode.step_start_times[0] - episode.t_start for episode in self.episodes
        ]
        other_inferences = [
            t2 - t1
            for episode in self.episodes
            for t1, t2 in zip(episode.step_end_times[:-1], episode.step_start_times[1:])
        ]
        return np.mean(first_inferences + other_inferences)


def number_of_better_final_beams(
    study_1: Study,
    study_2: Study,
) -> int:
    """
    Computer the number of times that the best MAE of a run in `study_1` is better than
    the best MAE of the same run in `study_2`.
    """
    assert study_1.are_problems_unique(), "The problems in study 1 are note unique."
    assert study_2.are_problems_unique(), "The problems in study 2 are note unique."

    study_1_idxs = sorted(study_1.problem_indicies())
    study_2_idxs = sorted(study_2.problem_indicies())
    assert study_1_idxs == study_2_idxs, "The studies do not cover the same problems."

    problem_idxs = study_1_idxs
    best_maes_1 = [
        study_1.get_episodes_by_problem(i)[0].best_mae() for i in problem_idxs
    ]
    best_maes_2 = [
        study_2.get_episodes_by_problem(i)[0].best_mae() for i in problem_idxs
    ]

    diff = np.array(best_maes_1) - np.array(best_maes_2)

    return sum(diff < 0)


def screen_extent(
    resolution: tuple[int, int], pixel_size: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Compute extent of a diagnostic screen for Matplotlib plotting."""
    return (
        -resolution[0] * pixel_size[0] / 2,
        resolution[0] * pixel_size[0] / 2,
        -resolution[1] * pixel_size[1] / 2,
        resolution[1] * pixel_size[1] / 2,
    )


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


def plot_best_mae_box(studies: list[Study], save_path: Optional[str] = None) -> None:
    """Box plot of best MAEs seen until the very end of the episodes."""
    combined_best_maes = []
    combined_names = []
    for study in studies:
        best_maes = [episode.best_mae() for episode in study.episodes]

        names = [study.name] * len(best_maes)

        combined_best_maes += best_maes
        combined_names += names

    plt.figure(figsize=(5, 0.6 * len(studies)))
    plt.title("Best MAEs")
    sns.boxplot(x=combined_best_maes, y=combined_names)
    plt.xscale("log")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_best_mae_diff_over_problem(
    study_1: Study,
    study_2: Study,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the differences of the best MAE achieved for each problem to see if certain
    problems stand out.
    """
    assert study_1.are_problems_unique(), "The problems in study 1 are note unique."
    assert study_2.are_problems_unique(), "The problems in study 2 are note unique."

    study_1_idxs = sorted(study_1.problem_indicies())
    study_2_idxs = sorted(study_2.problem_indicies())
    assert study_1_idxs == study_2_idxs, "The studies do not cover the same problems."

    problem_idxs = study_1_idxs
    best_maes_1 = [
        study_1.get_episodes_by_problem(i)[0].best_mae() for i in problem_idxs
    ]
    best_maes_2 = [
        study_2.get_episodes_by_problem(i)[0].best_mae() for i in problem_idxs
    ]

    diff = np.array(best_maes_1) - np.array(best_maes_2)

    plt.figure(figsize=(5, 3))
    plt.bar(problem_idxs, diff, label=f"{study_1.name} vs. {study_2.name}")
    plt.legend()
    plt.xlabel("Problem Index")
    plt.ylabel("Best MAE")

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_best_mae_over_time(
    studies: list[Study],
    threshold: Optional[float] = None,
    logarithmic: bool = False,
    title: Optional[str] = "Mean Best MAE Over Time",
    figsize: tuple[float, float] = (5, 3),
    study_name_str: str = "Study",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot mean best seen MAE over all episdoes over time. Optionally display a
    `threshold` line to mark measurement limit. Set `logarithmic` to `True` to log scale
    the y-axis.
    """
    dfs = []
    for study in studies:
        ds = [
            {
                "MAE (m)": episode.min_maes(),
                "Step": range(len(episode)),
                "Problem Index": episode.problem_index,
                study_name_str: study.name,
            }
            for episode in study.episodes
        ]
        df = pd.concat(pd.DataFrame(d) for d in ds)

        dfs.append(df)

    combined_df = pd.concat(dfs)

    # Convert unit to mm
    combined_df["MAE (μm)"] = combined_df["MAE (m)"] * 1e6

    plt.figure(figsize=figsize)
    if threshold is not None:
        plt.axhline(threshold, ls="--", color="lightsteelblue", label="Threshold")
    sns.lineplot(x="Step", y="MAE (μm)", hue=study_name_str, data=combined_df)
    plt.ylabel(r"MAE ($\mu$m)")
    plt.title(title)
    plt.xlim(0, None)
    if logarithmic:
        plt.yscale("log")
    else:
        plt.ylim(0, None)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_final_mae_box(
    studies: list[Study],
    title: Optional[str] = "Final MAEs",
    figsize: tuple[float, float] = (5, 3),
    save_path: Optional[str] = None,
) -> None:
    """
    Box plot of the final MAE that the algorithm stopped at (without returning to best
    seen).
    """
    combined_final_maes = []
    combined_names = []
    for study in studies:
        final_maes = [episode.final_mae() for episode in study.episodes]

        names = [study.name] * len(final_maes)

        combined_final_maes += final_maes
        combined_names += names

    combined_final_maes = np.array(combined_final_maes) * 1e6  # Convert to micro meters

    plt.figure(figsize=figsize)
    plt.title(title)
    sns.boxplot(x=combined_final_maes, y=combined_names)
    plt.xscale("log")
    plt.xlabel(r"MAE ($\mu$m)")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_mae_over_time(
    studies: list[Study],
    threshold: Optional[float] = None,
    logarithmic: bool = False,
    legend: bool = True,
    title: Optional[str] = "Mean MAE Over Time",
    figsize: tuple[float, float] = (5, 3),
    study_name_str: str = "Study",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot mean MAE of over episodes over time. Optionally display a `threshold` line to
    mark measurement limit. Set `logarithmic` to `True` to log scale the y-axis.
    """
    dfs = []
    for study in studies:
        ds = [
            {
                "MAE (m)": episode.maes(),
                "Step": range(len(episode)),
                "Problem Index": episode.problem_index,
                study_name_str: study.name,
            }
            for episode in study.episodes
        ]
        df = pd.concat(pd.DataFrame(d) for d in ds)

        dfs.append(df)

    combined_df = pd.concat(dfs)

    # Convert unit to mm
    combined_df["MAE (μm)"] = combined_df["MAE (m)"] * 1e6

    plt.figure(figsize=figsize)
    if threshold is not None:
        plt.axhline(threshold, ls="--", color="lightsteelblue", label="Threshold")
    sns.lineplot(
        x="Step", y="MAE (μm)", hue=study_name_str, data=combined_df, legend=legend
    )
    plt.title(title)
    plt.ylabel(r"MAE ($\mu$m)")
    plt.xlim(0, None)
    if logarithmic:
        plt.yscale("log")
    else:
        plt.ylim(0, None)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

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
    plt.gca().set_axisbelow(True)
    plt.xlabel("No. of steps")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_screen_image(
    image: np.ndarray,
    resolution: np.ndarray,
    pixel_size: np.ndarray,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: tuple[float, float] = (12, 10),
    save_path: Optional[Path] = None,
    vmax: Optional[float] = None,
    xlabel: bool = True,
    ylabel: bool = True,
) -> matplotlib.axes.Axes:
    """Plot an image of a diagnostic screen."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    image_float = image.astype("float")
    image_filtered = uniform_filter(image_float, size=10)
    image_nan = image_filtered.copy()
    image_nan[image_filtered < 10] = np.nan

    ax.imshow(
        image_nan,
        vmin=0,
        vmax=image_filtered.max() if vmax is None else vmax,
        aspect="equal",
        interpolation="none",
        extent=screen_extent(resolution, pixel_size * 1e6),
        cmap="rainbow",
    )
    if xlabel:
        ax.set_xlabel(r"($\mu$m)")
    if ylabel:
        ax.set_ylabel(r"($\mu$m)")

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)

    return ax


def plot_beam_parameters_on_screen(
    mu_x: float,
    sigma_x: float,
    mu_y: float,
    sigma_y: float,
    resolution: Optional[np.ndarray] = None,
    pixel_size: Optional[np.ndarray] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: tuple[float, float] = (12, 10),
    save_path: Optional[Path] = None,
    xlabel: bool = True,
    ylabel: bool = True,
    measurement_accuracy: float = 20e-6,
) -> matplotlib.axes.Axes:
    """
    Plot beam parameters indicated on an area representing the screen. Can be used as an
    overlay.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sigma_x = max(sigma_x, measurement_accuracy)
    sigma_y = max(sigma_y, measurement_accuracy)

    ax.axvline(mu_x * 1e6)
    ax.axhline(mu_y * 1e6)

    sigma_ellipse = Ellipse(
        xy=(mu_x * 1e6, mu_y * 1e6),
        width=6 * sigma_x * 1e6 / 2,
        height=6 * sigma_y * 1e6 / 2,
        facecolor="none",
        edgecolor="red",
        lw=1,
    )
    ax.add_patch(sigma_ellipse)

    if resolution is not None and pixel_size is not None:
        (left, right, bottom, top) = screen_extent(resolution, pixel_size * 1e6)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

    if xlabel:
        ax.set_xlabel(r"($\mu$m)")
    if ylabel:
        ax.set_ylabel(r"($\mu$m)")

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)

    return ax
