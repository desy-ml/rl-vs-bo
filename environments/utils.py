from gym import spaces
import numpy as np
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
        raise NotImplementedError()
    else:
        raise ValueError(f"There exists no beam parameter method \"{method}\"!")


def compute_beam_parameters_via_fwhm(img, pixel_size):
    parameters = np.empty(4)
    for axis in [0, 1]:
        profile = img.sum(axis=axis)
        minfiltered = minimum_filter1d(profile, size=5, mode="nearest")
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
