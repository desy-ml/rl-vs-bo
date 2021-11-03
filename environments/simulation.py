import cheetah
import numpy as np

from . import utils
from .ares import ARESlatticeStage3v1_9 as lattice


class ExperimentalArea:
    """Cheetah Simulation of the Experimental Area at ARES."""

    screen_resolution = (2448, 2040)
    pixel_size = (3.3198e-6, 2.4469e-6)

    def __init__(self, incoming="const", misalignments="none", measure_beam="us"):
        self._incoming_method = incoming
        if incoming == "problem":
            self.next_incoming = None
        self._misalignment_method = misalignments
        if misalignments == "problem":
            self.next_misalignments = None

        self._beam_parameter_method = measure_beam
        
        self._segment = cheetah.Segment.from_ocelot(lattice.cell, warnings=False, device="cpu").subcell("AREASOLA1", "AREABSCR1")
        self._segment.AREABSCR1.resolution = self.screen_resolution
        self._segment.AREABSCR1.pixel_size = self.pixel_size
        self._segment.AREABSCR1.is_active = True
        self._segment.AREABSCR1.binning = 4
    
    def reset(self):
        if self._incoming_method == "const":
            self._incoming = cheetah.ParameterBeam.from_parameters()
        elif self._incoming_method == "random":
            self._incoming = self._make_random_incoming()
        elif self._incoming_method == "problem":
            if self.next_incoming is None:
                raise ValueError("No next incoming beam has been set.")
            self._incoming = cheetah.ParameterBeam.from_parameters(**self.next_incoming)
            self.next_incoming = None
        else:
            raise ValueError(f"Invalid setting \"{self._incoming_method}\" for incoming method.")
        
        if self._misalignment_method == "none":
            self.misalignments = np.zeros(8)
        elif self._misalignment_method == "random":
            self.misalignments = np.random.uniform(-400e-6, 400e-6, size=8)
        elif self._misalignment_method == "problem":
            if self.next_misalignments is None:
                raise ValueError("No next misalignments have been set")
            self.misalignments = self.next_misalignments
            self.next_misalignments = None
        else:
            raise ValueError(f"Invalid setting \"{self._misalignment_method}\" for misalignment method.")

        self._run_simulation()
    
    def _make_random_incoming(self):
        return cheetah.ParameterBeam.from_parameters(
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
    
    @property
    def misalignments(self):
        return [
            self._segment.AREAMQZM1.misalignment[0],
            self._segment.AREAMQZM1.misalignment[1],
            self._segment.AREAMQZM2.misalignment[0],
            self._segment.AREAMQZM2.misalignment[1],
            self._segment.AREAMQZM3.misalignment[0],
            self._segment.AREAMQZM3.misalignment[1],
            self._segment.AREABSCR1.misalignment[0],
            self._segment.AREABSCR1.misalignment[1]
        ]

    @misalignments.setter
    def misalignments(self, value):
        self._segment.AREAMQZM1.misalignment = (value[0], value[1])
        self._segment.AREAMQZM2.misalignment = (value[2], value[3])
        self._segment.AREAMQZM3.misalignment = (value[4], value[5])
        self._segment.AREABSCR1.misalignment = (value[6], value[7])

    @property
    def actuators(self):
        return np.array([self._segment.AREAMQZM1.k1,
                         self._segment.AREAMQZM2.k1,
                         self._segment.AREAMQZM3.k1,
                         self._segment.AREAMCVM1.angle,
                         self._segment.AREAMCHM1.angle])
    
    @actuators.setter
    def actuators(self, values):
        self._segment.AREAMQZM1.k1, self._segment.AREAMQZM2.k1, self._segment.AREAMQZM3.k1 = values[:3]
        self._segment.AREAMCVM1.angle, self._segment.AREAMCHM1.angle = values[3:]

        self._run_simulation()
    
    def capture_clean_beam(self):
        return self._segment.AREABSCR1.reading
    
    def compute_beam_parameters(self):
        if self._beam_parameter_method == "direct":
            return self._read_beam_parameters_from_simulation()
        else:
            image = self.capture_clean_beam()
            return utils.compute_beam_parameters(
                image,
                self.pixel_size * self.binning,
                method=self._beam_parameter_method)
    
    def _read_beam_parameters_from_simulation(self):
        return np.array([
            self._segment.AREABSCR1.read_beam.mu_x,
            self._segment.AREABSCR1.read_beam.mu_y,
            self._segment.AREABSCR1.read_beam.sigma_x,
            self._segment.AREABSCR1.read_beam.sigma_y
        ])

    
    def _run_simulation(self):
        _ = self._segment(self._incoming)
    
    @property
    def binning(self):
        return self._segment.AREABSCR1.binning
