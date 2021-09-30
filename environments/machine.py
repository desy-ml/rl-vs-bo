import importlib
import os
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from . import simulation


pydoocs = importlib.import_module(os.getenv("EARLMCP", "dummypydoocs"))


class ARESEAMachine(simulation.ARESEACheetah):
    """Version of the ARES EA environment to interface with the real accelerator using PyDoocs."""

    actuator_channels = [
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/",
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/"
    ]

    def __init__(self):
        pass
    
    def reset(self, goal=None):
        if goal is not None:
            self.goal = goal
        else:
            self.goal = self.accelerator_observation_space["desired_goal"].sample()

        self.screen_data = self.read_screen()

        self.finished_steps = 0
        objective = self.compute_objective(
            self.observation["achieved_goal"],
            self.observation["desired_goal"]
        )
        self.history = [{
            "objective": objective,
            "reward": np.nan,
            "observation": self.observation,
            "action": np.full_like(self.action_space.high, np.nan)
        }]

        return self.observation2agent(self.observation)
    
    def render(self, mode="human"):
        fig = plt.figure("ARESEA-PyDoocs", figsize=(28,8))
        fig.clear()

        gs = gridspec.GridSpec(2, 4, wspace=0.35, hspace=0.3, figure=fig)

        ax_screen = fig.add_subplot(gs[0,0])
        self.plot_screen(ax_screen)

        ax_obs = fig.add_subplot(gs[0,1])
        self.plot_observations(ax_obs)

        ax_goal = fig.add_subplot(gs[1,1])
        self.plot_goals(ax_goal)
        
        sgs_act = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[1,2,1], subplot_spec=gs[:,2])
        ax_act = fig.add_subplot(sgs_act[1,0])
        self.plot_actions(ax_act)

        ax_rew = fig.add_subplot(gs[0,3])
        self.plot_rewards(ax_rew)
        
        ax_obj = fig.add_subplot(gs[1,3])
        self.plot_objective(ax_obj)

        if mode == "rgb_array":
            return mplfig_to_npimage(fig)
        if mode == "human":
            plt.show()
        else:
            raise ValueError(f"Invalid render mode \"{mode}\" (allowed: {self.metadata['render.modes']})")

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
        
        self.screen_data = self.read_screen()

    def read_screen(self):
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
            profile = self.screen_data.sum(axis=axis)
            minfiltered = minimum_filter1d(profile, size=5, mode="nearest")
            filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

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
