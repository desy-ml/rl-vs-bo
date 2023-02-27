from pathlib import Path

import cheetah
import numpy as np
import pytest
from gym.wrappers import RescaleAction
from stable_baselines3.common.env_checker import check_env

from backend import (
    BCCheetahBackend,
    BCDOOCSBackend,
    DLCheetahBackend,
    DLDOOCSBackend,
    EACheetahBackend,
    EADOOCSBackend,
    EAOcelotBackend,
    SHCheetahBackend,
    SHDOOCSBackend,
)
from environment import (
    BCTransverseTuning,
    DLTransverseTuning,
    EATransverseTuning,
    SHTransverseTuning,
)
from eval_bo_sim_quad_aligned import find_quad_aligned_incoming_beam_parameters
from trial import load_trials


@pytest.mark.parametrize(
    "backend_cls", [EACheetahBackend, EADOOCSBackend, EAOcelotBackend]
)
def test_ea_check_env(backend_cls):
    env = EATransverseTuning(backend=backend_cls(), w_beam=1.0, w_mu_x=1.0)
    env = RescaleAction(env, -1, 1)
    check_env(env)


@pytest.mark.parametrize("backend_cls", [BCCheetahBackend, BCDOOCSBackend])
def test_bc_check_env(backend_cls):
    env = BCTransverseTuning(backend=backend_cls(), w_beam=1.0, w_mu_x=1.0)
    env = RescaleAction(env, -1, 1)
    check_env(env)


@pytest.mark.parametrize("backend_cls", [DLCheetahBackend, DLDOOCSBackend])
def test_dl_check_env(backend_cls):
    env = DLTransverseTuning(backend=backend_cls(), w_beam=1.0, w_mu_x=1.0)
    env = RescaleAction(env, -1, 1)
    check_env(env)


@pytest.mark.parametrize("backend_cls", [SHCheetahBackend, SHDOOCSBackend])
def test_sh_check_env(backend_cls):
    env = SHTransverseTuning(backend=backend_cls(), w_beam=1.0, w_mu_x=1.0)
    env = RescaleAction(env, -1, 1)
    check_env(env)


def test_find_incoming_beam_for_zero_misalignment():
    trial = load_trials(Path("trials.yaml"))[0]
    trial.misalignments = np.zeros(8)

    new_incoming_beam = find_quad_aligned_incoming_beam_parameters(trial)

    assert all(new_incoming_beam[[1, 2, 3, 4]] == 0)


@pytest.mark.parametrize("direction", [-1, 1])
def test_find_incoming_beam_for_same_offset_misalignment(direction):
    trial = load_trials(Path("trials.yaml"))[0]
    trial.misalignments = direction * np.ones(8)

    new_incoming_beam = find_quad_aligned_incoming_beam_parameters(trial)

    correct_offset = all(new_incoming_beam[[1, 3]] == direction)
    no_transverse_momentum = all(new_incoming_beam[[2, 4]] == 0)
    assert correct_offset and no_transverse_momentum


@pytest.mark.parametrize("direction", [-1, 1])
def test_find_incoming_beam_for_increasing_offset_misalignment(direction):
    trial = load_trials(Path("trials.yaml"))[0]
    trial.misalignments = direction * np.array([1, 1, 2, 2, 3, 3, 0, 0])

    new_incoming_beam = find_quad_aligned_incoming_beam_parameters(trial)

    transverse_momentums = new_incoming_beam[[2, 4]]
    assert all(transverse_momentums > 0 if direction == 1 else transverse_momentums < 0)


def test_find_incoming_beam_slope_to_momentum():
    trial = load_trials(Path("trials.yaml"))[0]
    trial.misalignments = np.array(
        [
            200e-6,
            200e-6,
            200e-6 / 0.24 * 0.79,
            200e-6 / 0.24 * 0.79,
            200e-6 / 0.24 * 1.33,
            200e-6 / 0.24 * 1.33,
            0,
            0,
        ]
    )
    assumed_slope = 200e-6 / 0.24

    new_beam_parameters = find_quad_aligned_incoming_beam_parameters(trial)

    cheetah_beam = cheetah.ParameterBeam.from_parameters(
        energy=new_beam_parameters[0],
        mu_x=new_beam_parameters[1],
        mu_xp=new_beam_parameters[2],
        mu_y=new_beam_parameters[3],
        mu_yp=new_beam_parameters[4],
        sigma_x=new_beam_parameters[5],
        sigma_xp=new_beam_parameters[6],
        sigma_y=new_beam_parameters[7],
        sigma_yp=new_beam_parameters[8],
        sigma_s=new_beam_parameters[9],
        sigma_p=new_beam_parameters[10],
    )
    drifted_beam = cheetah.Drift(length=1.0)(cheetah_beam)

    offset_zero = np.isclose(cheetah_beam.mu_x, 0) and np.isclose(cheetah_beam.mu_y, 0)
    slope_matches_momentum = np.isclose(
        drifted_beam.mu_x, assumed_slope
    ) and np.isclose(drifted_beam.mu_y, assumed_slope)

    assert offset_zero and slope_matches_momentum
