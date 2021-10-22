import cheetah
import numpy as np

from .ares import ARESlatticeStage3v1_9 as lattice


class ExperimentalArea:
    """Cheetah Simulation of the Experimental Area at ARES."""

    screen_resolution = (2448, 2040)
    pixel_size = (3.3198e-6, 2.4469e-6)

    def __init__(self):
        self.segment = cheetah.Segment.from_ocelot(lattice.cell, warnings=False).subcell("AREASOLA1", "AREABSCR1")
        self.segment.AREABSCR1.resolution = self.screen_resolution
        self.segment.AREABSCR1.pixel_size = self.pixel_size
        self.segment.AREABSCR1.is_active = True

        self.segment.AREABSCR1.binning = 4

        self.randomize_incoming()
    
    def randomize_incoming(self):
        self.incoming = cheetah.ParameterBeam.from_parameters(
            mu_x=np.random.uniform(-3e-3, 3e-3),
            mu_y=np.random.uniform(-3e-4, 3e-4),
            mu_xp=np.random.uniform(-1e-4, 1e-4),
            mu_yp=np.random.uniform(-1e-4, 1e-4),
            sigma_x=np.random.uniform(0, 2e-3),
            sigma_y=np.random.uniform(0, 2e-3),
            sigma_xp=np.random.uniform(0, 1e-4),
            sigma_yp=np.random.uniform(0, 1e-4),
            sigma_s=np.random.uniform(0, 2e-3),
            sigma_p=np.random.uniform(0, 5e-3),
            energy=np.random.uniform(80e6, 160e6)
        )
        self._run_simulation()

    @property
    def actuators(self):
        return np.array([self.segment.AREAMQZM1.k1,
                         self.segment.AREAMQZM2.k1,
                         self.segment.AREAMQZM3.k1,
                         self.segment.AREAMCVM1.angle,
                         self.segment.AREAMCHM1.angle])
    
    @actuators.setter
    def actuators(self, values):
        self.segment.AREAMQZM1.k1, self.segment.AREAMQZM2.k1, self.segment.AREAMQZM3.k1 = values[:3]
        self.segment.AREAMCVM1.angle, self.segment.AREAMCHM1.angle = values[3:]

        self._run_simulation()
    
    def capture_clean_beam(self):
        return self.segment.AREABSCR1.reading
    
    def _run_simulation(self):
        _ = self.segment(self.incoming)
    
    @property
    def binning(self):
        return self.segment.AREABSCR1.binning
