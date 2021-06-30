from copy import deepcopy

import numpy as np
import ocelot as oc

from accelerator_environments import utils
import accelerator_environments.envs.ares.ARESlatticeStage3v1_9 as lattice
from accelerator_environments.envs.ares.aresea_v0 import ARESEA


class ARESEAOcelot(ARESEA):
    """Ocelot version of the ARES experimental area."""

    def __init__(self):
        super().__init__()

        self.cell = utils.subcell_of(lattice.cell, "AREASOLA1", "ARMRBSCR1")
        self.method = oc.MethodTM()
        self.method.global_method = oc.SecondTM
        self.lattice = oc.MagneticLattice(self.cell, method=self.method)

        self.areamqzm1 = lattice.areamqzm1
        self.areamqzm2 = lattice.areamqzm2
        self.areamqzm3 = lattice.areamqzm3
        self.areamcvm1 = lattice.areamcvm1
        self.areamchm1 = lattice.areamchm1

        self.incoming_particles = oc.generate_parray(nparticles=int(1e+5),
                                                     sigma_x=175e-6,
                                                     sigma_px=2e-7,
                                                     sigma_y=175e-6,
                                                     sigma_py=2e-7,
                                                     sigma_p=0.0,
                                                     chirp=0,
                                                     energy=0.1,
                                                     sigma_tau=0.0)

        self.screen_bin_edges = (np.linspace(-self.screen_resolution[0] * self.pixel_size[0] / 2,
                                             self.screen_resolution[0] * self.pixel_size[0] / 2,
                                             num=int(self.screen_resolution[0] / self.binning) + 1),
                                 np.linspace(-self.screen_resolution[1] * self.pixel_size[1] / 2,
                                             self.screen_resolution[1] * self.pixel_size[1] / 2,
                                             num=int(self.screen_resolution[1] / self.binning) + 1))
        
    def initial_actuator_values(self):
        """Return an action with the initial values for all actuators."""
        return np.zeros_like(self.optimization_space.high)
    
    def read_actuators(self):
        return np.array([self.areamqzm1.k1,
                         self.areamqzm2.k1,
                         self.areamqzm3.k1,
                         self.areamcvm1.angle,
                         self.areamchm1.angle])

    def write_actuators(self, values):
        """Set magents to the values in the given action."""
        self.areamqzm1.k1, self.areamqzm2.k1, self.areamqzm3.k1 = values[:3]
        self.areamcvm1.angle, self.areamchm1.angle = values[3:]
        
        self.lattice.update_transfer_maps()

    def read_screen(self):
        """Get pixel data from the screen."""
        navigator = oc.Navigator(self.lattice)
        _, self.particles = oc.track(self.lattice, deepcopy(self.incoming_particles), navigator,
                                     calc_tws=False, print_progress=False)
        
        image, _, _ = np.histogram2d(self.particles[:,0], self.particles[:,2],
                                     bins=self.screen_bin_edges)
        image = np.flipud(image.T)
                
        return image
    
    def extract_beam_parameters(self, screen_data):
        return np.array([self.particles.x().mean(),
                         self.particles.y().mean(),
                         self.particles.x().std(),
                         self.particles.y().std()])
