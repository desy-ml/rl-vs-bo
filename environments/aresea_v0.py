import gym
from gym.spaces import Box
from matplotlib import cm
import numpy as np

from accelerator_environments.envs.optimization import Optimizable


class ARESEA(gym.Env, Optimizable):
    """Ocelot Gym environment of the ARES experimental area."""

    metadata = {"render.modes": ["rgb_array"],
                "video.frames_per_second": 5}
    
    binning = 4
    screen_resolution = (2448, 2040)
    pixel_size = (3.5488e-6, 2.5003e-6)

    observation_space = Box(low=np.array([0] * 4 + [-1] * 5, dtype=np.float32),
                            high=np.ones(9, dtype=np.float32))
    action_space = Box(low=-1, high=1, shape=(5,))
    optimization_space = Box(low=-1, high=1, shape=(5,))
    
    actuator_scalars = np.array([10, 10, 10, 1e-3, 1e-3])
    beam_parameter_scalars = np.array([screen_resolution[0] * pixel_size[0] / 2, 
                                       screen_resolution[1] * pixel_size[1] / 2, 
                                       screen_resolution[0] * pixel_size[0] / 2, 
                                       screen_resolution[1] * pixel_size[1] / 2])
    
    goal = np.array([1e-5, 1e-5, 5e-5, 5e-5]) / beam_parameter_scalars
            
    def reset(self):
        values = self.initial_actuator_values()
        self.write_actuators(values)

        beam_parameters = self.beam_parameters()
        actuators = self.read_actuators()
        observation = np.concatenate([beam_parameters, actuators])

        self.previous_score = self.score(beam_parameters)
        self.finished_steps = 0

        self.history = [{"score": self.previous_score,
                         "reward": 0,
                         "observation": observation,
                         "action": np.zeros(5)}]

        return observation
    
    def initial_actuator_values(self):
        """Return an action with the initial values for all actuators."""
        return self.read_actuators()

    def step(self, action):
        self.write_actuators(self.read_actuators() + action * 0.1)

        beam_parameters = self.beam_parameters()
        actuators = self.read_actuators()
        observation = np.concatenate([beam_parameters, actuators])

        score = self.score(beam_parameters)
        reward = self.previous_score - score
        self.previous_score = score

        done = (np.abs(beam_parameters) < self.goal).all()

        self.finished_steps += 1
        self.history.append({"score": score,
                             "reward": reward,
                             "observation": observation,
                             "action": action})

        return observation, reward, done, {}
    
    def score(self, beam_parameters):
        return np.abs(beam_parameters).sum()

    def read_actuators(self):
        """Read the values currently set on the magnets."""
        return self.read_magnets() / self.actuator_scalars
    
    def read_magnets(self):
        raise NotImplementedError
    
    def write_actuators(self, values):
        """Set magents to the values in the given action."""
        self.write_magnets(values * self.actuator_scalars)
    
    def write_magnets(self, values):
        raise NotImplementedError

    def read_screen(self):
        """Get pixel data from the screen."""
        raise NotImplementedError

    def beam_parameters(self):
        self.screen_data = self.read_screen()

        parameters = np.empty(4)
        for axis in [0, 1]:
            profile = self.screen_data.sum(axis=axis)
            parameters[axis] = profile.argmax() * self.pixel_size[axis] - self.screen_resolution[axis] * self.pixel_size[axis] / 2
            half_values, = np.where(profile >= 0.5 * profile.max())
            parameters[axis+2] = (half_values[-1] - half_values[0]) * self.pixel_size[axis] / 2

        return parameters / self.beam_parameter_scalars
    
    def objective_function(self, actuators):
        """Computes the objective function for a given action."""
        self.write_actuators(actuators)
        beam_parameters = self.beam_parameters()
        score = self.score(beam_parameters)

        return score

    def render(self, mode="human", close=False):
        
        if mode=="rgb_array":
            scalar_mappable = cm.ScalarMappable(cmap="magma")
            rgba = scalar_mappable.to_rgba(self.screen_data)
            rgb = (rgba[:,:,:3] * 255).astype("uint8")
        
            return rgb
        else:
            raise ValueError(f"Invalid render mode \"{mode}\" (allowed: {self.metadata['render.modes']})")
