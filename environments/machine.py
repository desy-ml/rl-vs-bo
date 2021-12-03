import importlib
import logging
import os
import time

import numpy as np

from . import utils


pydoocs = importlib.import_module(os.getenv("EARLMCP", "dummypydoocs"))


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
        self._setup_logger()

    def reset(self):
        pass
    
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

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
        
        self.logger.debug(f"Setting actuators to {list(values)}")
        
        for channel, value in zip(self.actuator_channels[:3], values[:3]):
            pydoocs.write(channel + "STRENGTH.SP", value)
        for channel, value in zip(self.actuator_channels[3:], values[3:]):
            pydoocs.write(channel + "KICK_MRAD.SP", value * 1000)
        
        self._wait_for_magnets(self.actuator_channels)
    
    def capture_clean_beam(self, average=10):
        """Capture a clean (dark current removed) image of the beam."""
        self._wait_machine_okay()

        self.logger.debug("Capturing clean beam")

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
        self.logger.debug("Turning laser on")
        self._switch_cathode_laser(True)
    
    def _cathode_laser_off(self):
        self.logger.debug("Turning laser off")
        self._switch_cathode_laser(False)
    
    @property
    def binning(self):
        return (
            pydoocs.read(self.screen_channel + "BINNINGHORIZONTAL")["data"],
            pydoocs.read(self.screen_channel + "BINNINGVERTICAL")["data"]
        )
    
    def _wait_machine_okay(self, timeout=600):
        self.logger.debug("Checking machine okay")

        if self._error_count > 0:
            self.logger.warning("Waiting for machine okay")
            
            t1, t2 = time.time(), time.time()
            
            while self._error_count > 0 and t2 - t1 < timeout:
                time.sleep(30)
                t2 = time.time()
                self.logger.debug(f"Waiting for machine okay (timeout in {int(timeout-(t2-t1))} seconds)")
                
            if self._error_count > 0:
                self._go_to_safe_state()
                self.logger.error("Wait machine okay timed out -> machine set to safe state")
                raise Exception(f"Error count was above 0 for more than {timeout} seconds")
        
        self.logger.debug("Machine is okay")
                
    @property
    def _error_count(self):
        response = pydoocs.read("SINBAD.UTIL/MACHINE.STATE/ACCLXSISRV04._SVR/SVR.ERROR_COUNT")
        return response["data"]
    
    def _go_to_safe_state(self):
        self.logger.info("Going to safe state")
        self._switch_cathode_laser(False)
        self._zero_magnets()
    
    def _zero_magnets(self):
        for channel in self.actuator_channels[:3]:
            pydoocs.write(channel + "STRENGTH.SP", 0)
        for channel in self.actuator_channels[3:]:
            pydoocs.write(channel + "KICK_MRAD.SP", 0)
    
    def _wait_for_magnets(self, channels, timeout=180):
        self.logger.debug("Waiting for magnets")
        time.sleep(3.0)

        t1, t1a, t2 = time.time(), time.time(), time.time()
        while not self._are_magnets_ready(channels) and t2 - t1 < timeout:
            if t2 - t1a > 30:
                self.logger.debug(f"Waiting for magnets (timeout in {int(timeout-(t2-t1))} seconds)")
                t1a = time.time()
            
            t2 = time.time()
        
        if not self._are_magnets_ready(channels):
            self._recover_magnets(channels)
        
        self.logger.debug("Magnets are ready")
    
    def _recover_magnets(self, channels, timeout=120):
        self.logger.debug("Entering magnet recovery")

        self._wait_machine_okay()

        i = 0
        while not self._are_magnets_ready(channels) and i < 10:
            i += 1

            # Via zero
            self.logger.debug(f"Attemping magnet recovery via zero")
            broken = [channel for channel in channels if not self._is_ps_on(channel) or self._is_busy(channel)]
            self.logger.debug(f"Detected magnets requiring recovery: {broken}")

            for channel in broken:
                sp = pydoocs.read(f"{channel}CURRENT.SP")["data"]
                pydoocs.write(f"{channel}CURRENT.SP", 0)
                time.sleep(10)
                self._turn_on_ps(channel)

                t1, t2 = time.time(), time.time()
                while not self._are_magnets_ready(broken) and t2 - t1 < timeout:
                    time.sleep(30)
                    t2 = time.time()
                    self.logger.debug(f"Waiting for magnets to recover (timeout in {int(timeout-(t2-t1))} seconds)")

                if not self._is_ps_on(channel):
                    self.logger.debug("Magnet did not turn back on")
                    continue
                
                pydoocs.write(f"{channel}CURRENT.SP", sp)

                t1, t2 = time.time(), time.time()
                while not self._are_magnets_ready(broken) and t2 - t1 < timeout:
                    time.sleep(30)
                    t2 = time.time()
                    self.logger.debug(f"Waiting for magnets to recover (timeout in {int(timeout-(t2-t1))} seconds)")

        if self._are_magnets_ready(channels):
            self.logger.debug("Magnet recovery was successful")
        else:
            self._go_to_safe_state()
            self.logger.error("Magnet recovery failed -> machine set to safe state")
            raise Exception(f"Magnet setting timed out")
    
    def _is_busy(self, channel):
        return pydoocs.read(channel + "BUSY")["data"]
    
    def _is_ps_on(self, channel):
        return pydoocs.read(channel + "PS_ON")["data"]
    
    def _are_magnets_ready(self, channels):
        return all(self._is_ps_on(channel) and not self._is_busy(channel) for channel in channels)
    
    def _turn_on_ps(self, channel):
        self.logger.debug(f"Turning on power supply \"{channel}\"")
        pydoocs.write(channel + "PS_ON", 1)
        time.sleep(1)

    def _turn_off_ps(self, channel):
        self.logger.debug(f"Turning off power supply \"{channel}\"")
        pydoocs.write(channel + "PS_ON", 0)
        time.sleep(1)
    
    def _restart_ps(self, channel):
        self._turn_off_ps(channel)
        self._turn_on_ps(channel)

    def _wiggle(self, channel, delta):
        self.logger.debug(f"Wiggling \"{channel}\" by {delta}")

        current = pydoocs.read(channel)["data"]
        wiggle_value = current + delta
        pydoocs.write(channel, wiggle_value)
        time.sleep(1)
        pydoocs.write(channel, current)
