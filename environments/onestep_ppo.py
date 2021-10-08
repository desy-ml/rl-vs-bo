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
    

    def __init__(self, backend="simulation", random_incoming=False, random_initial=False, beam_parameter_method="us"):
        self.backend = backend
        self.random_incoming = random_incoming
        self.random_initial = random_initial
        self.beam_parameter_method = beam_parameter_method

        if self.backend == "simulation":
            self.accelerator = simulation.ExperimentalArea()
        elif self.backend == "machine":
            self.accelerator = machine.ExperimentalArea()
        else:
            raise ValueError(f"There is no \"{backend}\" backend!")
    
    def reset(self, desired=None):
        if self.random_incoming:
            self.accelerator.randomize_incoming()
        
        if self.random_initial:
            self.accelerator.actuators = self.action_space.sample()
        
        self.desired = desired if desired is not None else self.beam_parameter_space.sample()
        
        self._screen_data = self.accelerator.capture_clean_beam()
        self.achieved = self.compute_beam_parameters(self._screen_data)

        observation = np.concatenate([self.accelerator.actuators, self.desired, self.achieved])
        normalized_observation = self._normalize_observation(observation)
        
        return normalized_observation
    
    def step(self, action):
        self.accelerator.actuators = self._denormalize_action(action)

        self._screen_data = self.accelerator.capture_clean_beam()
        self.achieved = self.compute_beam_parameters(self._screen_data)
        objective = self._objective_fn(self.achieved, self.desired)

        observation = np.concatenate([self.accelerator.actuators, self.desired, self.achieved])
        normalized_observation = self._normalize_observation(observation)

        return normalized_observation, -objective, True, {}
    
    def _objective_fn(self, achieved, desired):
        offset = achieved - desired
        weights = np.array([1, 1, 2, 2])

        return np.log((weights * np.abs(offset)).sum())
    
    def _denormalize_action(self, normalized):
        return normalized * self.action_space.high
    
    def _normalize_observation(self, raw):
        scaler = np.concatenate([
            self.action_space.high,
            self.beam_parameter_space.high,
            self.beam_parameter_space.high
        ])
        return raw / scaler
    
    def compute_beam_parameters(self, image):
        if self.beam_parameter_method == "direct":
            return self._read_beam_parameters_from_simulation()
        else:
            return utils.compute_beam_parameters(
                image,
                self.accelerator.pixel_size*self.accelerator.binning,
                method=self.beam_parameter_method)
    
    def _read_beam_parameters_from_simulation(self):
        return np.array([
            self.accelerator.segment.AREABSCR1.read_beam.mu_x,
            self.accelerator.segment.AREABSCR1.read_beam.mu_y,
            self.accelerator.segment.AREABSCR1.read_beam.sigma_x,
            self.accelerator.segment.AREABSCR1.read_beam.sigma_y
        ])
