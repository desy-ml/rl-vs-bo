from collections import namedtuple
import importlib
import logging
import os
import time

import cv2
import gym
from gym import spaces
import numpy as np
from scipy import optimize
from scipy.ndimage import minimum_filter1d, uniform_filter1d

pydoocs = importlib.import_module(os.getenv("EARLMCP", "dummypydoocs"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


MagnetSettings = namedtuple(
    "MagnetSettings", ["AREAMQZM1", "AREAMQZM2", "AREAMCVM1", "AREAMQZM3", "AREAMCHM1"]
)
BeamParameters = namedtuple("BeamParameters", ["mu_x", "sigma_x", "mu_y", "sigma_y"])


class ARESEA(gym.Env):

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 2}

    observation_space = spaces.Dict(
        {
            "beam": spaces.Box(
                low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            ),
            "magnets": spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            ),
        }
    )

    action_space = spaces.Box(
        low=np.array([0, 0, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
        high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
    )

    def reset(self):
        self.set_magnets(MagnetSettings(0, 0, 0, 0, 0))

        beam_image = self.read_beam_image()
        pixel_size = self.read_pixel_size()
        binning = self.read_binning()
        self.initial_screen_beam = self.compute_beam_parameters(
            beam_image, pixel_size * binning
        )

        magnet_readbacks = self.read_magnets()

        observation = {
            "beam": np.array(
                [
                    self.initial_screen_beam.mu_x,
                    self.initial_screen_beam.sigma_x,
                    self.initial_screen_beam.mu_y,
                    self.initial_screen_beam.sigma_y,
                ],
                dtype=np.float32,
            ),
            "magnets": np.array(
                [
                    magnet_readbacks.AREAMQZM1,
                    -magnet_readbacks.AREAMQZM2,  # NOTE the sign here
                    magnet_readbacks.AREAMCVM1,
                    magnet_readbacks.AREAMQZM3,
                    magnet_readbacks.AREAMCHM1,
                ],
                dtype=np.float32,
            ),
        }

        # Save these as common knowledge for render to use, so it doesn't have to be recomputed
        self.current_beam_image = beam_image
        self.current_beam = self.initial_screen_beam
        self.current_magnet_settings = magnet_readbacks

        return observation

    def step(self, action):
        # Perform action
        new_magnet_settings = MagnetSettings(
            AREAMQZM1=action[0],
            AREAMQZM2=-action[1],  # NOTE the sign here
            AREAMCVM1=action[2],
            AREAMQZM3=action[3],
            AREAMCHM1=action[4],
        )
        self.set_magnets(new_magnet_settings)

        beam_image = self.read_beam_image()
        pixel_size = self.read_pixel_size()
        binning = self.read_binning()
        current_beam = self.compute_beam_parameters(beam_image, pixel_size * binning)

        magnet_readbacks = self.read_magnets()

        # Build observation
        observation = {
            "beam": np.array(
                [
                    current_beam.mu_x,
                    current_beam.sigma_x,
                    current_beam.mu_y,
                    current_beam.sigma_y,
                ],
                dtype=np.float32,
            ),
            "magnets": np.array(
                [
                    magnet_readbacks.AREAMQZM1,
                    -magnet_readbacks.AREAMQZM2,  # NOTE the sign here
                    magnet_readbacks.AREAMCVM1,
                    magnet_readbacks.AREAMQZM3,
                    magnet_readbacks.AREAMCHM1,
                ],
                dtype=np.float32,
            ),
        }

        on_screen_reward = 0  # TODO Placeholder
        mu_x_reward = -abs(current_beam.mu_x / self.initial_screen_beam.mu_x)
        sigma_x_reward = -current_beam.sigma_x / self.initial_screen_beam.sigma_x
        mu_y_reward = -abs(current_beam.mu_y / self.initial_screen_beam.mu_y)
        sigma_y_reward = -current_beam.sigma_y / self.initial_screen_beam.sigma_y
        # aspect_ratio_reward = - abs(current_beam.sigma_x - current_beam.sigma_y)

        # TODO: Maybe add aspect ratio term
        reward = (
            1 * on_screen_reward
            + 1 * mu_x_reward
            + 1 * sigma_x_reward
            + 1 * mu_y_reward
            + 1 * sigma_y_reward
        )
        reward = float(reward)

        # Figure out if reach good enough beam (done)
        done = bool(np.all(observation["beam"] < 3.3198e-6))

        # Put together info
        info = {
            "on_screen_reward": 0 * on_screen_reward,
            "mu_x_reward": 1 * mu_x_reward,
            "sigma_x_reward": 1 * sigma_x_reward,
            "mu_y_reward": 1 * mu_y_reward,
            "sigma_y_reward": 1 * sigma_y_reward,
        }

        # Save these as common knowledge for render to use, so it doesn't have to be recomputed
        self.current_beam_image = beam_image
        self.current_beam = current_beam
        self.current_magnet_settings = magnet_readbacks

        return observation, reward, done, info

    def render(self, mode="human"):
        assert mode == "rgb_array" or mode == "human"

        # Read screen image and make 8-bit RGB
        img = self.current_beam_image
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)

        # Draw beam ellipse
        beam = self.current_beam
        pixel_size = self.read_pixel_size() * self.read_binning()
        resolution = self.read_resolution() / self.read_binning()
        e_pos_x = int(beam.mu_x / pixel_size[0] + resolution[0] / 2)
        e_width_x = int(beam.sigma_x / pixel_size[0])
        e_pos_y = int(-beam.mu_y / pixel_size[1] + resolution[1] / 2)
        e_width_y = int(beam.sigma_y / pixel_size[1])
        red = (0, 0, 255)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, red, 2
        )

        # Adjust aspect ration
        new_width = int(img.shape[1] * pixel_size[0] / pixel_size[1])
        img = cv2.resize(img, (new_width, img.shape[0]))

        # Add magnet values
        magnet_settings = self.current_magnet_settings
        padding = np.full(
            (int(img.shape[0] * 0.18), img.shape[1], 3), fill_value=255, dtype=np.uint8
        )
        img = np.vstack([img, padding])
        black = (0, 0, 0)
        img = cv2.putText(
            img,
            f"Q1={magnet_settings.AREAMQZM1:.2f}",
            (15, 545),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        img = cv2.putText(
            img,
            f"Q2={magnet_settings.AREAMQZM2:.2f}",
            (215, 545),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        img = cv2.putText(
            img,
            f"CV={magnet_settings.AREAMCVM1*1e3:.2f}",
            (415, 545),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        img = cv2.putText(
            img,
            f"Q3={magnet_settings.AREAMQZM3:.2f}",
            (615, 545),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        img = cv2.putText(
            img,
            f"CH={magnet_settings.AREAMCHM1*1e3:.2f}",
            (15, 585),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )

        if mode == "human":
            cv2.imshow("ARES EA", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def read_magnets(self):
        return MagnetSettings(
            AREAMQZM1=pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.RBV")[
                "data"
            ],
            AREAMQZM2=pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.RBV")[
                "data"
            ],
            AREAMCVM1=pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.RBV")[
                "data"
            ]
            / 1000,
            AREAMQZM3=pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.RBV")[
                "data"
            ],
            AREAMCHM1=pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.RBV")[
                "data"
            ]
            / 1000,
        )

    def set_magnets(self, settings, wait=True):
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.SP", settings.AREAMQZM1
        )
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.SP", settings.AREAMQZM2
        )
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.SP", settings.AREAMCVM1 * 1000
        )
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.SP", settings.AREAMQZM3
        )
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.SP", settings.AREAMCHM1 * 1000
        )

        # Wait until magnets have reached their setpoints

        time.sleep(3.0)  # Wait for magnets to realise they received a command

        magnets = ["AREAMQZM1", "AREAMQZM2", "AREAMCVM1", "AREAMQZM3", "AREAMCHM1"]

        are_busy = [True] * 5
        are_ps_on = [True] * 5
        while any(are_busy) or not all(are_ps_on):
            are_busy = [
                pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/BUSY")["data"]
                for magnet in magnets
            ]
            are_ps_on = [
                pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/PS_ON")["data"]
                for magnet in magnets
            ]

    def read_beam_image(self, average=5):
        # Laser off
        self.set_cathode_laser(False)
        background_images = self.capture_interval(n=average, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        self.set_cathode_laser(True)
        beam_images = self.capture_interval(n=average, dt=0.1)
        median_beam = np.median(beam_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16 - 1)
        flipped = np.flipud(removed)

        return flipped

    def capture_interval(self, n, dt):
        images = []
        for _ in range(n):
            images.append(self.read_screen())
            time.sleep(dt)
        return np.array(images)

    def read_screen(self):
        return pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ")["data"]

    def set_cathode_laser(self, setto):
        """Sets the bool switch of the cathode laser event to setto and waits a second."""
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)

    def compute_beam_parameters(self, img, pixel_size):
        parameters = {}
        for axis, direction in zip([0, 1], ["x", "y"]):
            projection = img.sum(axis=axis)
            minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
            filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

            (half_values,) = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2
            else:
                fwhm_pixel = 42  # TODO: Figure out what to do with these
                center_pixel = 42

            parameters[f"mu_{direction}"] = (
                center_pixel - len(filtered) / 2
            ) * pixel_size[axis]
            parameters[f"sigma_{direction}"] = fwhm_pixel / 2.355 * pixel_size[axis]

        parameters["mu_y"] = -parameters["mu_y"]

        return BeamParameters(**parameters)

    def read_pixel_size(self):
        return np.array(
            (
                abs(
                    pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/X.POLY_SCALE")[
                        "data"
                    ][2]
                ),
                abs(
                    pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/Y.POLY_SCALE")[
                        "data"
                    ][2]
                ),
            )
        )

    def read_resolution(self):
        return np.array((2464, 2056))  # TODO Actually read from pydoocs

    def read_binning(self):
        return np.array(
            (
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL")[
                    "data"
                ],
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL")[
                    "data"
                ],
            )
        )


def compute_beam_parameters(img, pixel_size, method="us"):
    if method == "us":
        return compute_beam_parameters_via_fwhm(img, pixel_size)
    elif method == "willi":
        return compute_beam_parameters_via_gaussianfit(img, pixel_size)
    else:
        raise ValueError(f'There exists no beam parameter method "{method}"!')


def compute_beam_parameters_via_fwhm(img, pixel_size):
    parameters = np.empty(4)
    for axis in [0, 1]:
        projection = img.sum(axis=axis)
        minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
        filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

        (half_values,) = np.where(filtered >= 0.5 * filtered.max())

        if len(half_values) > 0:
            fwhm_pixel = half_values[-1] - half_values[0]
            center_pixel = half_values[0] + fwhm_pixel / 2
        else:
            fwhm_pixel = 42  # TODO: Figure out what to do with these
            center_pixel = 42

        parameters[axis] = (center_pixel - len(filtered) / 2) * pixel_size[axis]
        parameters[axis + 2] = fwhm_pixel / 2.355 * pixel_size[axis]

    parameters[1] = -parameters[1]

    return parameters


def compute_beam_parameters_via_gaussianfit(img, pixel_size):
    def gaussian(x, a, mu, sigma):
        return np.abs(a) * np.exp(-(((x - mu) / sigma) ** 2) / 2)

    parameters = np.empty(4)
    for axis in [0, 1]:
        projection = img.sum(axis=axis)
        minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
        filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

        pixel_centers = np.arange(len(filtered)) + 0.5

        result, _ = optimize.curve_fit(gaussian, pixel_centers, filtered)
        _, mu_pixel, sigma_pixel = result

        parameters[axis] = (mu_pixel - len(filtered) / 2) * pixel_size[axis]
        parameters[axis + 2] = sigma_pixel * pixel_size[axis]

    parameters[1] = -parameters[1]

    return parameters
