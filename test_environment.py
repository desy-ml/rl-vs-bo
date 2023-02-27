from pathlib import Path

import cheetah
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


@pytest.mark.parametrize("trial_index", range(300))
def test_simulation_beam_to_quadrupole_alignment(trial_index):
    trial = load_trials(Path("trials.yaml"))[trial_index]
    backend = EACheetahBackend(
        misalignment_mode="constant", misalignment_values=trial.misalignments
    )
    backend.reset()
    segment = backend.segment

    bad_beam_parameters = trial.incoming_beam
    bad_incoming_beam = cheetah.ParameterBeam.from_parameters(
        energy=bad_beam_parameters[0],
        mu_x=bad_beam_parameters[1],
        mu_xp=bad_beam_parameters[2],
        mu_y=bad_beam_parameters[3],
        mu_yp=bad_beam_parameters[4],
        sigma_x=bad_beam_parameters[5],
        sigma_xp=bad_beam_parameters[6],
        sigma_y=bad_beam_parameters[7],
        sigma_yp=bad_beam_parameters[8],
        sigma_s=bad_beam_parameters[9],
        sigma_p=bad_beam_parameters[10],
    )
    good_beam_parameters = find_quad_aligned_incoming_beam_parameters(trial)
    good_incoming_beam = cheetah.ParameterBeam.from_parameters(
        energy=good_beam_parameters[0],
        mu_x=good_beam_parameters[1],
        mu_xp=good_beam_parameters[2],
        mu_y=good_beam_parameters[3],
        mu_yp=good_beam_parameters[4],
        sigma_x=good_beam_parameters[5],
        sigma_xp=good_beam_parameters[6],
        sigma_y=good_beam_parameters[7],
        sigma_yp=good_beam_parameters[8],
        sigma_s=good_beam_parameters[9],
        sigma_p=good_beam_parameters[10],
    )

    segment.AREAMQZM1.k1 = 0.0
    segment.AREAMQZM2.k1 = 0.0
    segment.AREAMQZM3.k1 = 0.0
    segment.AREABSCR1.is_active = False

    # Before beam-to-quad alignment
    beam_0 = segment(bad_incoming_beam)
    segment.AREAMQZM1.k1 = 10.0
    beam_1 = segment(bad_incoming_beam)
    segment.AREAMQZM1.k1 = 0.0
    segment.AREAMQZM2.k1 = 10.0
    beam_2 = segment(bad_incoming_beam)
    segment.AREAMQZM2.k1 = 0.0
    segment.AREAMQZM3.k1 = 10.0
    beam_3 = segment(bad_incoming_beam)
    segment.AREAMQZM3.k1 = 0.0

    bad_movement = (
        abs(beam_1.mu_x - beam_0.mu_x)
        + abs(beam_2.mu_x - beam_0.mu_x)
        + abs(beam_3.mu_x - beam_0.mu_x)
        + abs(beam_1.mu_y - beam_0.mu_y)
        + abs(beam_2.mu_y - beam_0.mu_y)
        + abs(beam_3.mu_y - beam_0.mu_y)
    )

    # After beam-to-quad alignment
    beam_0 = segment(good_incoming_beam)
    segment.AREAMQZM1.k1 = 10.0
    beam_1 = segment(good_incoming_beam)
    segment.AREAMQZM1.k1 = 0.0
    segment.AREAMQZM2.k1 = 10.0
    beam_2 = segment(good_incoming_beam)
    segment.AREAMQZM2.k1 = 0.0
    segment.AREAMQZM3.k1 = 10.0
    beam_3 = segment(good_incoming_beam)
    segment.AREAMQZM3.k1 = 0.0

    good_movement = (
        abs(beam_1.mu_x - beam_0.mu_x)
        + abs(beam_2.mu_x - beam_0.mu_x)
        + abs(beam_3.mu_x - beam_0.mu_x)
        + abs(beam_1.mu_y - beam_0.mu_y)
        + abs(beam_2.mu_y - beam_0.mu_y)
        + abs(beam_3.mu_y - beam_0.mu_y)
    )

    assert good_movement <= bad_movement
