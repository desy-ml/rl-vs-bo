import time

import numpy as np
import pydoocs

from accelerator_environments import utils
from accelerator_environments.envs.ares.aresea_v0 import ARESEA


class ARESEAMachine(ARESEA):
    """Environment to interface with the actual experimental area at the ARES machine."""
    
    channels = ["SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/",
                "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/",
                "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/",
                "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/",
                "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/"]
    
    def __init__(self):
        super().__init__()

        pydoocs.write("SINBAD.DIAG/CAMERA/AR.MR.BSC.R.1/BINNINGHORIZONTAL", self.binning)
        pydoocs.write("SINBAD.DIAG/CAMERA/AR.MR.BSC.R.1/BINNINGVERTICAL", self.binning)

    def read_magnets(self):
        data = []
        for channel in self.channels[:3]:
            data.append(pydoocs.read(channel + "STRENGTH.RBV")["data"])
        for channel in self.channels[3:]:
            data.append(pydoocs.read(channel + "KICK_MRAD.RBV")["data"] / 1000)
        
        return np.array(data)

    def write_magnets(self, action):
        """Set the magents on ARES (the actual machine) in the experimental area."""

        for channel, value in zip(self.channels[:3], action[:3]):
            pydoocs.write(channel + "STRENGTH.SP", value)
        for channel, value in zip(self.channels[3:], action[3:]):
            pydoocs.write(channel + "KICK_MRAD.SP", value * 1000)
        
        while any(pydoocs.read(channel + "BUSY")["data"] for channel in self.channels):
            time.sleep(0.25)
    
    def read_screen(self):
        """Get pixel data from the screen."""

        channel = "SINBAD.DIAG/CAMERA/AR.MR.BSC.R.1/IMAGE_EXT_ZMQ"

        self.switch_cathode_laser(False)
        background = self.capture(10, channel).mean(axis=0)
        
        self.switch_cathode_laser(True)
        beam = self.capture(10, channel).mean(axis=0)

        image = beam - background
        
        return image
    
    def capture(self, n, channel):
        images = []
        for _ in range(n):
            images.append(pydoocs.read(channel)["data"])
            time.sleep(0.1)
        return np.array(images)
    
    def switch_cathode_laser(self, setto):
        """Sets the bool switch of the cathode laser event to setto and waits a second."""
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)
