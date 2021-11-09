import gym
from gym import spaces
import numpy as np

from . import machine, simulation, utils


class ARESEAOptimization(gym.Env):
    """Variant of the ARES EA environment that uses absolute rather than relative actions."""

    action_space = spaces.Box(
        low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32),  # TODO: Quad limits +/-72
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

    target_delta = np.array([5e-6] * 4)     # TODO: Set to somethind like pixel accuracy
    
    def __init__(self, backend="simulation", backendargs={}):
        if backend == "simulation":
            self.backend = simulation.ExperimentalArea(**backendargs)
        elif backend == "machine":
            self.backend = machine.ExperimentalArea(**backendargs)
        else:
            raise ValueError(f"Backend {backend} is not supported!")

        self.next_initial = None
        self.next_desired = None
    
    def reset(self):
        self.backend.reset()
        
        if isinstance(self.next_initial, str) and self.next_initial == "stay":
            pass
        elif self.next_initial is not None:
            self.backend.actuators = self.next_initial
        else:
            self.backend.actuators = self.action_space.sample()
        self.next_initial = None

        self.desired = self.next_desired if self.next_desired is not None else self.beam_parameter_space.sample()
        self.next_desired = None
        
        self.achieved = self.backend.compute_beam_parameters()

        observation = np.concatenate([self.backend.actuators, self.desired, self.achieved])
        
        objective = self._objective_fn(self.achieved, self.desired)
        self.history = [{
            "objective": objective,
            "reward": np.nan,
            "observation": observation,
            "action": np.full_like(self.action_space.high, np.nan)
        }]

        return observation
    
    def step(self, action):
        self.backend.actuators = action

        self.achieved = self.backend.compute_beam_parameters()
        objective = self._objective_fn(self.achieved, self.desired)

        observation = np.concatenate([self.backend.actuators, self.desired, self.achieved])
        done = (np.abs(self.achieved - self.desired) < self.target_delta).all()

        return observation, objective, done, {}
    
    def _objective_fn(self, achieved, desired):
        return ((achieved - desired)**2).mean()
