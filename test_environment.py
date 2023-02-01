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
    DLTransverseTuning,
    EATransverseTuning,
    MRTransverseTuning,
    SHTransverseTuning,
)


@pytest.mark.parametrize(
    "backend_cls", [EACheetahBackend, EADOOCSBackend, EAOcelotBackend]
)
def test_ea_check_env(backend_cls):
    env = EATransverseTuning(backend=backend_cls(), w_beam=1.0, w_mu_x=1.0)
    env = RescaleAction(env, -1, 1)
    check_env(env)


@pytest.mark.parametrize("backend_cls", [BCCheetahBackend, BCDOOCSBackend])
def test_bc_check_env(backend_cls):
    env = MRTransverseTuning(backend=backend_cls(), w_beam=1.0, w_mu_x=1.0)
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
