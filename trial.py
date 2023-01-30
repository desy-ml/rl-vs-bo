from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Trial:
    """Reperesents conditions for a trial run in the ARES EA."""

    target_beam: np.ndarray
    incoming_beam: np.ndarray
    misalignments: np.ndarray
    initial_magnets: np.ndarray


def load_trials(filepath: Path) -> list[Trial]:
    """Load a set of trials from a `.yaml` file."""
    with open(filepath, "r") as f:
        raw = yaml.full_load(f.read())

    converted = []
    for i in sorted(raw.keys()):
        raw_trial = raw[i]

        target_beam = target_beam_from_dictionary(raw_trial["target"])
        incoming_beam = incoming_beam_from_dictionary(raw_trial["incoming"])
        misalignments = misalignments_from_dictionary(raw_trial["misalignments"])
        inital_magnets = initial_magnets_from_dictionary(raw_trial["initial"])

        converted_trial = Trial(
            target_beam, incoming_beam, misalignments, inital_magnets
        )
        converted.append(converted_trial)

    return converted


def target_beam_from_dictionary(raw: dict) -> np.ndarray:
    """
    Read a dictionary describing a target beam to a correctly arranged `np.ndarray`.
    """
    return np.array(
        [
            raw["mu_x"],
            raw["sigma_x"],
            raw["mu_y"],
            raw["sigma_y"],
        ]
    )


def incoming_beam_from_dictionary(raw: dict) -> np.ndarray:
    """
    Read a dictionary describing an incoming beam to a correctly arranged `np.ndarray`.
    """
    return np.array(
        [
            raw["energy"],
            raw["mu_x"],
            raw["mu_xp"],
            raw["mu_y"],
            raw["mu_yp"],
            raw["sigma_x"],
            raw["sigma_xp"],
            raw["sigma_y"],
            raw["sigma_yp"],
            raw["sigma_s"],
            raw["sigma_p"],
        ]
    )


def misalignments_from_dictionary(raw: dict) -> np.ndarray:
    """
    Read a dictionary describing misalignments to a correctly arranged `np.ndarray`.
    """
    return np.array(
        [
            raw["q1_x"],
            raw["q1_y"],
            raw["q2_x"],
            raw["q2_y"],
            raw["q3_x"],
            raw["q3_y"],
            raw["screen_x"],
            raw["screen_y"],
        ]
    )


def initial_magnets_from_dictionary(raw: dict) -> np.ndarray:
    """
    Read a dictionary describing initial magnet settings to a correctly arranged
    `np.ndarray`.
    """
    return np.array(
        [
            raw["q1"],
            raw["q2"],
            raw["cv"],
            raw["q3"],
            raw["ch"],
        ]
    )
