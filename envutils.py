import gym
from gym import spaces
import numpy as np
from scipy import optimize
from scipy.ndimage import minimum_filter1d, uniform_filter1d


def combine_spaces(*args):
    "Combines `gym.spaces.Box` spaces into one space."
    assert all(isinstance(space, spaces.Box) for space in args)
    return spaces.Box(
        low=np.concatenate([space.low for space in args]),
        high=np.concatenate([space.high for space in args])
    )


def compute_beam_parameters(img, pixel_size, method="us"):
    if method == "us":
        return compute_beam_parameters_via_fwhm(img, pixel_size)
    elif method == "willi":
        return compute_beam_parameters_via_gaussianfit(img, pixel_size)
    else:
        raise ValueError(f"There exists no beam parameter method \"{method}\"!")


def compute_beam_parameters_via_fwhm(img, pixel_size):
    parameters = np.empty(4)
    for axis in [0, 1]:
        projection = img.sum(axis=axis)
        minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
        filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

        half_values, = np.where(filtered >= 0.5 * filtered.max())

        if len(half_values) > 0:
            fwhm_pixel = half_values[-1] - half_values[0]
            center_pixel = half_values[0] + fwhm_pixel / 2
        else:
            fwhm_pixel = 42     # TODO: Figure out what to do with these
            center_pixel = 42

        parameters[axis] = (center_pixel - len(filtered) / 2) * pixel_size[axis]
        parameters[axis+2] = fwhm_pixel / 2.355 * pixel_size[axis]
        
    parameters[1] = -parameters[1]

    return parameters


def compute_beam_parameters_via_gaussianfit(img, pixel_size):
    def gaussian(x, a, mu, sigma):
        return np.abs(a) * np.exp(-((x - mu) / sigma)**2 / 2)

    parameters = np.empty(4)
    for axis in [0, 1]:
        projection = img.sum(axis=axis)
        minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
        filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

        pixel_centers = np.arange(len(filtered)) + 0.5

        result, _ = optimize.curve_fit(gaussian, pixel_centers, filtered)
        _, mu_pixel, sigma_pixel = result

        parameters[axis] = (mu_pixel - len(filtered) / 2) * pixel_size[axis]
        parameters[axis+2] = sigma_pixel * pixel_size[axis]
        
    parameters[1] = -parameters[1]

    return parameters


class ResetActuators(gym.Wrapper):

    def reset(self, **kwargs):
        self.env.unwrapped.next_initial = np.zeros(self.env.action_space.shape)
        return self.env.reset(**kwargs)


class ResetActuatorsToDFD(gym.Wrapper):

    def __init__(self, env, k1=10):
        super().__init__(env)
        self.k1 = k1

    def reset(self, **kwargs):
        self.env.unwrapped.next_initial = np.array([10, -10, 10, 0, 0], dtype="float")
        return self.env.reset(**kwargs)


class ResetActuatorsToRandom(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.unwrapped.next_initial = self.env.actuator_space.sample()
        return self.env.reset(**kwargs)
