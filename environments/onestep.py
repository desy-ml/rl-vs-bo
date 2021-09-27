import cheetah
import gym
from gym import spaces
import numpy as np

from . import ARESlatticeStage3v1_9 as lattice


class ARESEAOneStep(gym.Env):
    """Variant of the ARES EA environment that uses absolute rather than relative actions."""

    action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5+4+4,))

    actuator_space = spaces.Box(
        low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32),
        high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32)
    )
    goal_space = spaces.Box(
        low=np.array([-2e-3, -2e-3, 0, 0], dtype=np.float32),
        high=np.array([2e-3, 2e-3, 5e-4, 5e-4], dtype=np.float32)
    )
    
    def __init__(self):
        super().__init__()
        
        cell = cheetah.utils.subcell_of(lattice.cell, "AREASOLA1", "AREABSCR1")

        self.segment = cheetah.Segment.from_ocelot(cell)

        self.segment.AREABSCR1.resolution = (2448, 2040)
        self.segment.AREABSCR1.pixel_size = (3.3198e-6, 2.4469e-6)
        self.segment.AREABSCR1.is_active = True

        self.segment.AREABSCR1.binning = 4
    
    def reset(self, incoming=None, initial_actuators=None, desired=None):
        if incoming is None:
            self.incoming = cheetah.Beam.make_random(
                n=int(1e5),
                mu_x=np.random.uniform(-3e-3, 3e-3),
                mu_y=np.random.uniform(-3e-4, 3e-4),
                mu_xp=np.random.uniform(-1e-4, 1e-4),
                mu_yp=np.random.uniform(-1e-4, 1e-4),
                sigma_x=np.random.uniform(0, 2e-3),
                sigma_y=np.random.uniform(0, 2e-3),
                sigma_xp=np.random.uniform(0, 1e-4),
                sigma_yp=np.random.uniform(0, 1e-4),
                sigma_s=np.random.uniform(0, 2e-3),
                sigma_p=np.random.uniform(0, 5e-3),
                energy=np.random.uniform(80e6, 160e6)
            )
        else:
            self.incoming = incoming
            
        if initial_actuators is None:
            self.initial_actuators = self.actuator_space.sample()
        else:
            self.initial_actuators = initial_actuators
        
        if desired is None:
            self.desired = self.goal_space.sample()
        else:
            self.desired = desired

        achieved = self._track(self.initial_actuators)

        observation = np.concatenate([self.initial_actuators, self.desired, achieved])
        normalized_observation = self._normalize_observation(observation)
        
        return normalized_observation
    
    def step(self, action):
        actuators = self._denormalize_action(action)

        achieved = self._track(actuators)
        objective = self._objective_fn(achieved, self.desired)

        observation = np.concatenate([action, self.desired, achieved])
        normalized_observation = self._normalize_observation(observation)

        return normalized_observation, -objective, True, {}
    
    def _track(self, actuators):
        self.segment.AREAMQZM1.k1, self.segment.AREAMQZM2.k1, self.segment.AREAMQZM3.k1 = actuators[:3]
        self.segment.AREAMCVM1.angle, self.segment.AREAMCHM1.angle = actuators[3:]

        _ = self.segment(self.incoming)
        
        return np.array([
            self.segment.AREABSCR1.read_beam.mu_x,
            self.segment.AREABSCR1.read_beam.mu_y,
            self.segment.AREABSCR1.read_beam.sigma_x,
            self.segment.AREABSCR1.read_beam.sigma_y
        ])
    
    def _objective_fn(self, achieved, desired):
        offset = achieved - desired
        weights = np.array([1, 1, 2, 2])

        return np.log((weights * np.abs(offset)).sum())
    
    def _denormalize_action(self, normalized):
        return normalized * self.actuator_space.high
    
    def _normalize_observation(self, raw):
        scaler = np.concatenate([
            self.actuator_space.high,
            self.goal_space.high,
            self.goal_space.high
        ])
        return raw / scaler
