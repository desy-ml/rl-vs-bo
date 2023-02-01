import pytest
from stable_baselines3.common.env_checker import check_env

from backend import CheetahBackend, DOOCSBackend, OcelotBackend
from environment import EATransverseTuning


@pytest.mark.parametrize("backend_cls", [CheetahBackend, DOOCSBackend, OcelotBackend])
def test_check_env(backend_cls):
    env = EATransverseTuning(backend=backend_cls(), w_beam=1.0, w_mu_x=1.0)
    check_env(env)
