import pytest
from gym.wrappers import RescaleAction
from stable_baselines3.common.env_checker import check_env

from backend import EACheetahBackend, EADOOCSBackend, EAOcelotBackend
from environment import EATransverseTuning


@pytest.mark.parametrize(
    "backend_cls", [EACheetahBackend, EADOOCSBackend, EAOcelotBackend]
)
def test_check_env(backend_cls):
    env = EATransverseTuning(backend=backend_cls(), w_beam=1.0, w_mu_x=1.0)
    env = RescaleAction(env, -1, 1)
    check_env(env)
