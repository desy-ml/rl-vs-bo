from datetime import datetime
import importlib
import logging
import os
from pathlib import Path
import time

import numpy as np

from . import utils
import toolkit


pydoocs = importlib.import_module(os.getenv("EARLMCP", "dummypydoocs"))


# Setup logging (to console)
timestamp = lambda: datetime.now().strftime("%y%m%d_%H%M%S")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)-7.7s: %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
path = os.getcwd() # Check path and directory before running
folder = 'logs'
Path(folder).mkdir(parents=True, exist_ok=True)
directory = os.path.join(path, folder)
description  = 'machine_backend'
logpath = os.path.join(directory,f'{timestamp()}_{description}.log')

file_handler = logging.FileHandler(logpath)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)-7.7s: %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class ExperimentalArea:
    """Interface to the Experimental Area at ARES."""

    actuator_channels = [
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/"
    ]
    screen_channel = "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/"

    screen_resolution = np.array([2464, 2056])
    pixel_size = np.array([3.3198e-6, 2.4469e-6])

    def __init__(self, measure_beam="us"):
        self._beam_parameter_method = measure_beam

    def reset(self):
        pass

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
        self._wait_machine_okay()
        
        logger.debug(f"Setting actuators to {values}")
        
        for channel, value in zip(self.actuator_channels[:3], values[:3]):
            pydoocs.write(channel + "STRENGTH.SP", value)
        for channel, value in zip(self.actuator_channels[3:], values[3:]):
            pydoocs.write(channel + "KICK_MRAD.SP", value * 1000)
        
        time.sleep(3.0)
        i = 0
        while any(pydoocs.read(channel + "BUSY")["data"] or not pydoocs.read(channel + "PS_ON")["data"] for channel in self.actuator_channels):
            time.sleep(0.25)
            i += 1
            if i > 180:
                for channel in self.actuator_channels:
                    if not pydoocs.read(channel + "PS_ON")["data"]:
                        pydoocs.write(channel + "PS_ON", 1)
                        time.sleep(1)
                    elif pydoocs.read(channel + "BUSY")["data"]:
                        pydoocs.write(channel + "PS_ON", 0)
                        time.sleep(1)
                        pydoocs.write(channel + "PS_ON", 1)
                        time.sleep(1)
            if i > 360:
                self._go_to_safe_state()
                logger.error("Magnet setting timed out and could not be recvoered -> machine set to safe state")
                toolkit.send_mail(
                    "Magnet setting timed out and could not be recvoered -> machine set to safe state",
                    ["oliver.stein@desy.de","jan.kaiser@desy.de","florian.burkart@desy.de"]
                )
                raise Exception(f"Magnet setting timed out")

    
    def capture_clean_beam(self, average=10):
        """Capture a clean (dark current removed) image of the beam."""
        self._wait_machine_okay()

        # Laser off
        self._cathode_laser_off()
        background_images = self._capture_interval(n=average, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        self._cathode_laser_on()
        beam_images = self._capture_interval(n=average, dt=0.1)
        median_beam = np.median(beam_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16-1)
        flipped = np.flipud(removed)

        self.last_beam_image = flipped
        
        return flipped
    
    def compute_beam_parameters(self):
        image = self.capture_clean_beam()
        return utils.compute_beam_parameters(
            image,
            self.pixel_size * self.binning,
            method=self._beam_parameter_method)
    
    def capture_screen(self):
        return pydoocs.read(self.screen_channel + "IMAGE_EXT_ZMQ")["data"]
    
    def _capture_interval(self, n, dt):
        images = []
        for _ in range(n):
            images.append(self.capture_screen())
            time.sleep(dt)
        return np.array(images)

    def _capture(self, n, channel):
        images = []
        for _ in range(n):
            images.append(pydoocs.read(channel)["data"].astype("float64"))
            time.sleep(0.1)
        return np.array(images)
    
    def _switch_cathode_laser(self, setto):
        """Sets the bool switch of the cathode laser event to setto and waits a second."""
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)
    
    def _cathode_laser_on(self):
        self._switch_cathode_laser(True)
    
    def _cathode_laser_off(self):
        self._switch_cathode_laser(False)
    
    @property
    def binning(self):
        return (
            pydoocs.read(self.screen_channel + "BINNINGHORIZONTAL")["data"],
            pydoocs.read(self.screen_channel + "BINNINGVERTICAL")["data"]
        )
    
    def _wait_machine_okay(self):
        timeout = 600
        if self._error_count > 0:
            logger.warning("Waiting for machine okay")
            i = 0
            while self._error_count > 0:
                time.sleep(1)
                i += 1
                logger.warning(f"Shutdown in {timeout-i} seconds")
                
                if i > timeout:
                    self._go_to_safe_state()
                    logger.error("Wait machine okay timed out -> machine set to safe state")
                    toolkit.send_mail(
                        "MSK-IPC AA: Wait machine okay timed out -> machine set to safe state",
                        ["oliver.stein@desy.de","jan.kaiser@desy.de","florian.burkart@desy.de"]
                    )
                    raise Exception(f"Error count was above 0 for more than {timeout} seconds")
                
    @property
    def _error_count(self):
        response = pydoocs.read("SINBAD.UTIL/MACHINE.STATE/ACCLXSISRV04._SVR/SVR.ERROR_COUNT")
        return response["data"]
    
    def _go_to_safe_state(self):
        self._switch_cathode_laser(False)
        self._zero_magnets()
    
    def _zero_magnets(self):
        for channel in self.actuator_channels[:3]:
            pydoocs.write(channel + "STRENGTH.SP", 0)
        for channel in self.actuator_channels[3:]:
            pydoocs.write(channel + "KICK_MRAD.SP", 0)
        