import gym
from gym import spaces
import numpy as np

from . import machine, simulation, utils


class ARESEAOneStep(gym.Env):
    """Variant of the ARES EA environment that uses absolute rather than relative actions."""

    action_space = spaces.Box(
        low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32),
        high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32)
    )
    beam_parameter_space = spaces.Box(
        low=np.array([-2e-3, -2e-3, 0, 0], dtype=np.float32),
        high=np.array([2e-3, 2e-3, 5e-4, 5e-4], dtype=np.float32)
    )
    observation_space = utils.combine_spaces(
        action_space,
        beam_parameter_space,
        beam_parameter_space
    )
    
    def __init__(self, backend="simulation", initial="none", backendargs={}):
        if backend == "simulation":
            self.backend = simulation.ExperimentalArea(**backendargs)
        elif backend == "machine":
            self.backend = machine.ExperimentalArea(**backendargs)
        else:
            raise ValueError(f"Backend {backend} is not supported!")

        self._initial_method = initial
    
    def reset(self, desired=None):
        self.backend.reset()
        
        if self._initial_method == "none":
            pass
        elif self._initial_method == "reset":
            self.backend.actuators = np.zeros(5)
        elif self._initial_method == "random":
            self.backend.actuators = self.actuator_space.sample()
        
        self.desired = desired if desired is not None else self.beam_parameter_space.sample()
        self.achieved = self.backend.compute_beam_parameters()

        observation = np.concatenate([self.backend.actuators, self.desired, self.achieved])
        
        return observation
    
    def step(self, action):
        self.backend.actuators = action

        self.achieved = self.backend.compute_beam_parameters()
        objective = self._objective_fn(self.achieved, self.desired)

        observation = np.concatenate([self.backend.actuators, self.desired, self.achieved])

        return observation, -objective, True, {}
    
    def _objective_fn(self, achieved, desired):
        offset = achieved - desired
        weights = np.array([1, 1, 2, 2])

        return np.log((weights * np.abs(offset)).sum())
