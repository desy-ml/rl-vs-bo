from gym.spaces import Box
import joss
from matplotlib.patches import Ellipse
import numpy as np

from accelerator_environments import utils
import accelerator_environments.envs.ares.ARESlatticeStage3v1_9 as lattice
import accelerator_environments.envs.ares.aresea_v1_joss as v1


class ARESEAJOSS(v1.ARESEAJOSS):
    """ARESEA version using a JOSS simulation as its backend."""
    
    pixel_size = (3.3198e-6, 2.4469e-6)

    observation_space = Box(low=np.array([-4e-3, -4e-3,    0,    0,   0, -30, -30, -30, -3e-3, -3e-3], dtype=np.float32),
                            high=np.array([4e-3,  4e-3, 4e-4, 4e-4, 1e5,  30,  30,  30,  3e-3,  3e-3], dtype=np.float32))
    action_space = Box(low=observation_space.low[-5:] * 0.1,
                       high=observation_space.high[-5:] * 0.1)
    optimization_space = Box(low=observation_space.low[-5:],
                             high=observation_space.high[-5:])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cell = utils.subcell_of(lattice.cell, "AREASOLA1", "AREABSCR1")
        self.segment = joss.Segment.from_ocelot(cell)
        self.segment.AREABSCR1.resolution = self.screen_resolution
        self.segment.AREABSCR1.pixel_size = self.pixel_size
        self.segment.AREABSCR1.binning = self.binning
        self.segment.AREABSCR1.is_active = True
        
    def read_screen(self):
        _ = self.segment(self.incoming)
        image = self.segment.AREABSCR1.reading

        return image
    
    @property
    def beam_parameters(self):
        if self.simulate_screen:
            return super().beam_parameters
        else:
            _ = self.screen_data    # TODO: This is a hack for now
            parameters = np.array([self.segment.AREABSCR1.read_beam.mu_x,
                                   self.segment.AREABSCR1.read_beam.mu_y,
                                   self.segment.AREABSCR1.read_beam.sigma_x,
                                   self.segment.AREABSCR1.read_beam.sigma_y]) 
            
            shifted = parameters - self.target

            return shifted
    
    def plot_screen(self, ax):
        ax.set_title("Screen")
        screen_extent = (-self.screen_resolution[0] * self.pixel_size[0] / 2, self.screen_resolution[0] * self.pixel_size[0] / 2,
                         -self.screen_resolution[1] * self.pixel_size[1] / 2, self.screen_resolution[1] * self.pixel_size[1] / 2)
        ax.imshow(self.screen_data, cmap="magma", interpolation="None", extent=screen_extent)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        target_mu_x, target_mu_y, target_sigma_x, target_sigma_y = self.target
        target_ellipse = Ellipse((target_mu_x,target_mu_y), target_sigma_x, target_sigma_y, fill=False, color="white")
        ax.add_patch(target_ellipse)

        read_mu_x, read_mu_y, read_sigma_x, read_sigma_y = self.target + self.beam_parameters
        read_ellipse = Ellipse((read_mu_x,read_mu_y), read_sigma_x, read_sigma_y, fill=False, color="deepskyblue", linestyle="--")
        ax.add_patch(read_ellipse)

        true_mu_x, true_mu_y, true_sigma_x, true_sigma_y = [self.segment.AREABSCR1.read_beam.mu_x,
                                                            self.segment.AREABSCR1.read_beam.mu_y,
                                                            self.segment.AREABSCR1.read_beam.sigma_x,
                                                            self.segment.AREABSCR1.read_beam.sigma_y]
        true_ellipse = Ellipse((true_mu_x,true_mu_y), true_sigma_x, true_sigma_y, fill=False, color="lime", linestyle="--")
        ax.add_patch(true_ellipse)
