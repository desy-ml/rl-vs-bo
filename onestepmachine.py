import importlib
import os
import time

from gym import spaces
import numpy as np
from scipy.ndimage import uniform_filter1d

from onestep import Simulation


pydoocs = importlib.import_module(os.getenv("EARLMCP", "dummypydoocs"))


class Machine(Simulation):

    screen_resolution = (2448, 2040)
    pixel_size = (3.3198e-6, 2.4469e-6)

    actuator_space = spaces.Box(
        low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32),
        high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32)
    )
    goal_space = spaces.Box(
        low=np.array([-2e-3, -2e-3, 0, 0], dtype=np.float32),
        high=np.array([2e-3, 2e-3, 5e-4, 5e-4], dtype=np.float32)
    )

    actuator_channels = [
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/"
    ]

    def __init__(self):
        pass

    def reset(self, desired=None):
        if desired is None:
            self.desired = self.goal_space.sample()
        else:
            self.desired = desired

        self._screen_data = self._read_screen()
        self.achieved = self.beam_parameters

        observation = np.concatenate([self.actuators, self.desired, self.achieved])

        return observation
    
    def track(self, actuators):
        self.actuators = actuators
        self._screen_data = self._read_screen()
        self.achieved = self.beam_parameters
                
        return self.achieved
    
    @property
    def actuators(self):
        data = []
        for channel in self.actuator_channels[:3]:
            data.append(pydoocs.read(channel + "STRENGTH.RBV")["data"])
        for channel in self.actuator_channels[3:]:
            data.append(pydoocs.read(channel + "KICK_MRAD.RBV")["data"] / 1000)
        
        return np.array(data)
    
    @actuators.setter
    def actuators(self, values):
        """Set the magents on ARES (the actual machine) in the experimental area."""
        for channel, value in zip(self.actuator_channels[:3], values[:3]):
            pydoocs.write(channel + "STRENGTH.SP", value)
        for channel, value in zip(self.actuator_channels[3:], values[3:]):
            pydoocs.write(channel + "KICK_MRAD.SP", value * 1000)
        
        time.sleep(3.0)
        while any(pydoocs.read(channel + "BUSY")["data"] or not pydoocs.read(channel + "PS_ON")["data"] for channel in self.actuator_channels):
            time.sleep(0.25)
        
        self._screen_data = self._read_screen()
    
    def _read_screen(self):
        """Get pixel data from the screen."""

        channel = "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ"

        self.switch_cathode_laser(False)
        self.backgrounds = self.capture(10, channel)
        self.background = np.median(self.backgrounds, axis=0)
        
        self.switch_cathode_laser(True)
        self.beams = self.capture(10, channel)
        self.beam = np.median(self.beams, axis=0)

        clean = self.beam - self.background
        image = clean.clip(0, 2**16-1)

        flipped = np.flipud(image)

        self.binning = (
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL")["data"],
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL")["data"]
        )
        
        return flipped

    def capture(self, n, channel):
        images = []
        for _ in range(n):
            images.append(pydoocs.read(channel)["data"].astype("float64"))
            time.sleep(0.1)
        return np.array(images)
    
    def switch_cathode_laser(self, setto):
        """Sets the bool switch of the cathode laser event to setto and waits a second."""
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)
    
    @property
    def beam_parameters(self):
        parameters = np.empty(4)
        for axis in [0, 1]:
            profile = self._screen_data.sum(axis=axis)
            # minfiltered = minimum_filter1d(profile, size=5, mode="nearest")
            filtered = uniform_filter1d(profile, size=5, mode="nearest")

            half_values, = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2
            else:
                fwhm_pixel = 42     # TODO: Figure out what to do with these
                center_pixel = 42

            parameters[axis] = (center_pixel - len(filtered) / 2) * self.pixel_size[axis] * self.binning[axis]
            parameters[axis+2] = fwhm_pixel / 2.355 * self.pixel_size[axis] * self.binning[axis]
            
        parameters[1] = -parameters[1]

        return parameters
