import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import cheetah
import dummypydoocs as pydoocs
import numpy as np
import ocelot as oc
from gym import spaces
from ocelot.cpbd.beam import generate_parray
from scipy.ndimage import minimum_filter1d, uniform_filter1d

import ARESlatticeStage3v1_9 as ares_oc
from ARESlatticeStage3v1_9 import (
    areabscr1,
    areamchm1,
    areamcvm1,
    areamqzm1,
    areamqzm2,
    areamqzm3,
    areasola1,
)
from ARESlatticeStage3v1_9 import cell as ares_lattice
from ARESlatticeStage3v1_9 import (
    drift_areamchm1,
    drift_areamcvm1,
    drift_areamqzm1,
    drift_areamqzm2,
    drift_areamqzm3,
    drift_areasola1,
)


class TransverseTuningBaseBackend(ABC):
    """Abstract class for a backend imlementation of the ARES Experimental Area."""

    @abstractmethod
    def is_beam_on_screen(self) -> bool:
        """
        Return `True` when the beam is on the screen and `False` when it isn't.

        Override with backend-specific imlementation. Must be implemented!
        """
        pass

    def setup(self) -> None:
        """
        Prepare the accelerator for use with the environment. Should mostly be used for
        setting up simulations.

        Override with backend-specific imlementation. Optional.
        """
        pass

    @abstractmethod
    def get_magnets(self) -> np.ndarray:
        """
        Return the magnet values as a NumPy array in order as the magnets appear in the
        accelerator.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    @abstractmethod
    def set_magnets(self, magnets: np.ndarray) -> None:
        """
        Set the magnets to the given values.

        The argument `magnets` will be passed as a NumPy array in the order the magnets
        appear in the accelerator.

        When applicable, this method should block until the magnet values are acutally
        set!

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Code that should set the accelerator up for a new episode. Run when the `reset`
        is called.

        Mostly meant for simulations to switch to a new incoming beam / misalignments or
        simular things.

        Override with backend-specific imlementation. Optional.
        """
        pass

    def update(self) -> None:
        """
        Update accelerator metrics for later use. Use this to run the simulation or
        cache the beam image.

        Override with backend-specific imlementation. Optional.
        """
        pass

    @abstractmethod
    def get_beam_parameters(self) -> np.ndarray:
        """
        Get the beam parameters measured on the diagnostic screen as NumPy array grouped
        by dimension (e.g. mu_x, sigma_x, mu_y, sigma_y).

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def get_incoming_parameters(self) -> np.ndarray:
        """
        Get all physical beam parameters of the incoming beam as NumPy array in order
        energy, mu_x, mu_xp, mu_y, mu_yp, sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s,
        sigma_p.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_misalignments(self) -> np.ndarray:
        """
        Get misalignments of the quadrupoles and the diagnostic screen as NumPy array in
        order AREAMQZM1.misalignment.x, AREAMQZM1.misalignment.y,
        AREAMQZM2.misalignment.x, AREAMQZM2.misalignment.y, AREAMQZM3.misalignment.x,
        AREAMQZM3.misalignment.y, AREABSCR1.misalignment.x, AREABSCR1.misalignment.y.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_screen_image(self) -> np.ndarray:
        """
        Retreive the beam image as a 2-dimensional NumPy array.

        Note that if reading the beam image is expensive, it is best to cache the image
        in the `update_accelerator` method and the read the cached variable here.

        Ideally, the pixel values should look somewhat similar to the 12-bit values from
        the real screen camera.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    @abstractmethod
    def get_binning(self) -> np.ndarray:
        """
        Return binning currently set on the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    @abstractmethod
    def get_screen_resolution(self) -> np.ndarray:
        """
        Return (binned) resolution of the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    @abstractmethod
    def get_pixel_size(self) -> np.ndarray:
        """
        Return the (binned) size of the area on the diagnostic screen covered by one
        pixel as NumPy array [x, y].

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def get_info(self) -> dict:
        """
        Return a dictionary of aditional info from the accelerator backend, e.g.
        incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}


class EACheetahBackend(TransverseTuningBaseBackend):
    """Cheetah simulation backend to the ARES Experimental Area."""

    def __init__(
        self,
        incoming_mode: str = "random",
        incoming_values: Optional[np.ndarray] = None,
        max_misalignment: float = 5e-4,
        misalignment_mode: str = "random",
        misalignment_values: Optional[np.ndarray] = None,
    ) -> None:
        self.incoming_mode = incoming_mode
        self.incoming_values = incoming_values
        self.max_misalignment = max_misalignment
        self.misalignment_mode = misalignment_mode
        self.misalignment_values = misalignment_values

        # Set up domain randomisation spaces
        self.incoming_beam_space = spaces.Box(
            low=np.array(
                [
                    80e6,
                    -1e-3,
                    -1e-4,
                    -1e-3,
                    -1e-4,
                    1e-5,
                    1e-6,
                    1e-5,
                    1e-6,
                    1e-6,
                    1e-4,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3],
                dtype=np.float32,
            ),
        )
        self.misalignment_space = spaces.Box(
            low=-self.max_misalignment, high=self.max_misalignment, shape=(8,)
        )

        # Create particle simulation
        self.segment = cheetah.Segment.from_ocelot(
            ares_lattice, warnings=False, device="cpu"
        ).subcell("AREASOLA1", "AREABSCR1")
        self.segment.AREABSCR1.resolution = (2448, 2040)
        self.segment.AREABSCR1.pixel_size = (3.3198e-6, 2.4469e-6)
        self.segment.AREABSCR1.binning = 4
        self.segment.AREABSCR1.is_active = True

    def is_beam_on_screen(self) -> bool:
        screen = self.segment.AREABSCR1
        beam_position = np.array([screen.read_beam.mu_x, screen.read_beam.mu_y])
        limits = np.array(screen.resolution) / 2 * np.array(screen.pixel_size)
        return np.all(np.abs(beam_position) < limits)

    def get_magnets(self) -> np.ndarray:
        return np.array(
            [
                self.segment.AREAMQZM1.k1,
                self.segment.AREAMQZM2.k1,
                self.segment.AREAMCVM1.angle,
                self.segment.AREAMQZM3.k1,
                self.segment.AREAMCHM1.angle,
            ]
        )

    def set_magnets(self, magnets: np.ndarray) -> None:
        self.segment.AREAMQZM1.k1 = magnets[0]
        self.segment.AREAMQZM2.k1 = magnets[1]
        self.segment.AREAMCVM1.angle = magnets[2]
        self.segment.AREAMQZM3.k1 = magnets[3]
        self.segment.AREAMCHM1.angle = magnets[4]

    def reset(self) -> None:
        # New domain randomisation
        if self.incoming_mode == "constant":
            incoming_parameters = self.incoming_values
        elif self.incoming_mode == "random":
            incoming_parameters = self.incoming_beam_space.sample()
        else:
            raise ValueError(f'Invalid value "{self.incoming_mode}" for incoming_mode')
        self.incoming = cheetah.ParameterBeam.from_parameters(
            energy=incoming_parameters[0],
            mu_x=incoming_parameters[1],
            mu_xp=incoming_parameters[2],
            mu_y=incoming_parameters[3],
            mu_yp=incoming_parameters[4],
            sigma_x=incoming_parameters[5],
            sigma_xp=incoming_parameters[6],
            sigma_y=incoming_parameters[7],
            sigma_yp=incoming_parameters[8],
            sigma_s=incoming_parameters[9],
            sigma_p=incoming_parameters[10],
        )

        if self.misalignment_mode == "constant":
            misalignments = self.misalignment_values
        elif self.misalignment_mode == "random":
            misalignments = self.misalignment_space.sample()
        else:
            raise ValueError(
                f'Invalid value "{self.misalignment_mode}" for misalignment_mode'
            )
        self.segment.AREAMQZM1.misalignment = misalignments[0:2]
        self.segment.AREAMQZM2.misalignment = misalignments[2:4]
        self.segment.AREAMQZM3.misalignment = misalignments[4:6]
        self.segment.AREABSCR1.misalignment = misalignments[6:8]

    def update(self) -> None:
        self.segment(self.incoming)

    def get_beam_parameters(self) -> np.ndarray:
        return np.array(
            [
                self.segment.AREABSCR1.read_beam.mu_x,
                self.segment.AREABSCR1.read_beam.sigma_x,
                self.segment.AREABSCR1.read_beam.mu_y,
                self.segment.AREABSCR1.read_beam.sigma_y,
            ]
        )

    def get_incoming_parameters(self) -> np.ndarray:
        # Parameters of incoming are typed out to guarantee their order, as the
        # order would not be guaranteed creating np.array from dict.
        return np.array(
            [
                self.incoming.energy,
                self.incoming.mu_x,
                self.incoming.mu_xp,
                self.incoming.mu_y,
                self.incoming.mu_yp,
                self.incoming.sigma_x,
                self.incoming.sigma_xp,
                self.incoming.sigma_y,
                self.incoming.sigma_yp,
                self.incoming.sigma_s,
                self.incoming.sigma_p,
            ]
        )

    def get_misalignments(self) -> np.ndarray:
        return np.array(
            [
                self.segment.AREAMQZM1.misalignment[0],
                self.segment.AREAMQZM1.misalignment[1],
                self.segment.AREAMQZM2.misalignment[0],
                self.segment.AREAMQZM2.misalignment[1],
                self.segment.AREAMQZM3.misalignment[0],
                self.segment.AREAMQZM3.misalignment[1],
                self.segment.AREABSCR1.misalignment[0],
                self.segment.AREABSCR1.misalignment[1],
            ],
            dtype=np.float32,
        )

    def get_screen_image(self) -> np.ndarray:
        # Screen image to look like real image by dividing by goodlooking number and
        # scaling to 12 bits)
        return self.segment.AREABSCR1.reading / 1e9 * 2**12

    def get_binning(self) -> np.ndarray:
        return np.array(self.segment.AREABSCR1.binning)

    def get_screen_resolution(self) -> np.ndarray:
        return np.array(self.segment.AREABSCR1.resolution) / self.get_binning()

    def get_pixel_size(self) -> np.ndarray:
        return np.array(self.segment.AREABSCR1.pixel_size) * self.get_binning()

    def get_info(self) -> dict:
        return {
            "incoming_beam": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
        }


class EAOcelotBackend(TransverseTuningBaseBackend):
    """Backend simulating the ARES EA in Ocelot."""

    def __init__(
        self,
        incoming_mode: str = "random",
        incoming_values: Optional[np.ndarray] = None,
        max_misalignment: float = 5e-4,
        misalignment_mode: str = "random",
        misalignment_values: Optional[np.ndarray] = None,
        include_space_charge: bool = True,
        charge: float = 1e-12,  # in C
        nparticles: int = int(1e5),
        unit_step: float = 0.01,  # tracking step in [m]
    ) -> None:
        self.incoming_mode = incoming_mode
        self.incoming_values = incoming_values
        self.max_misalignment = max_misalignment
        self.misalignment_mode = misalignment_mode
        self.misalignment_values = misalignment_values

        self.screen_resolution = (2448, 2040)
        self.screen_pixel_size = (3.3198e-6, 2.4469e-6)
        self.binning = 1

        # Set up domain randomisation spaces
        self.incoming_beam_space = spaces.Box(
            low=np.array(
                [
                    80e6,
                    -1e-3,
                    -1e-4,
                    -1e-3,
                    -1e-4,
                    1e-5,
                    1e-6,
                    1e-5,
                    1e-6,
                    1e-6,
                    1e-4,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3],
                dtype=np.float32,
            ),
        )
        self.misalignment_space = spaces.Box(
            low=-self.max_misalignment, high=self.max_misalignment, shape=(8,)
        )

        self.include_space_charge = include_space_charge
        self.charge = charge
        self.nparticles = nparticles
        self.unit_step = unit_step

        # Initialize Tracking method
        self.method = oc.MethodTM()
        self.method.global_method = oc.SecondTM
        self.unit_step = unit_step

        self.sc = oc.SpaceCharge()
        self.sc.nmesh_xyz = [32, 32, 32]
        self.sc.step = 1

        # Build lattice
        self.cell = (
            areasola1,
            drift_areasola1,
            areamqzm1,
            drift_areamqzm1,
            areamqzm2,
            drift_areamqzm2,
            areamcvm1,
            drift_areamcvm1,
            areamqzm3,
            drift_areamqzm3,
            areamchm1,
            drift_areamchm1,
            areabscr1,
        )

        self.lattice = oc.MagneticLattice(
            self.cell, start=areasola1, stop=areabscr1, method=self.method
        )

    def is_beam_on_screen(self) -> bool:
        beam_position = self.get_beam_parameters()[[0, 2]]
        limits = np.array(self.screen_resolution) / 2 * np.array(self.screen_pixel_size)
        return np.all(np.abs(beam_position) < limits)

    def get_magnets(self) -> np.ndarray:
        return np.array(
            [
                self.lattice[areamqzm1].k1,
                self.lattice[areamqzm2].k1,
                self.lattice[areamcvm1].angle,
                self.lattice[areamqzm3].k1,
                self.lattice[areamchm1].angle,
            ]
        )

    def set_magnets(self, magnets: np.ndarray) -> None:
        self.lattice[areamqzm1].k1 = magnets[0]
        self.lattice[areamqzm2].k1 = magnets[1]
        self.lattice[areamcvm1].angle = magnets[2]
        self.lattice[areamqzm3].k1 = magnets[3]
        self.lattice[areamqzm1].angle = magnets[4]
        self.lattcie = self.lattice.update_transfer_maps()

    def reset(self) -> None:
        # New domain randomisation
        if self.incoming_mode == "constant":
            incoming_parameters = self.incoming_values
        elif self.incoming_mode == "random":
            incoming_parameters = self.incoming_beam_space.sample()
        else:
            raise ValueError(f'Invalid value "{self.incoming_mode}" for incoming_mode')
        self.incoming = generate_parray(
            sigma_x=incoming_parameters[5],
            sigma_px=incoming_parameters[6],
            sigma_y=incoming_parameters[7],
            sigma_py=incoming_parameters[8],
            sigma_tau=incoming_parameters[9],
            sigma_p=incoming_parameters[10],
            energy=incoming_parameters[0] / 1e9,
            nparticles=self.nparticles,
            charge=self.charge,
        )
        # Set mu_x, mu_xp, mu_y, mu_yp
        self.incoming.rparticles[0] += incoming_parameters[1]  # mu_x
        self.incoming.rparticles[1] += incoming_parameters[2]  # mu_xp
        self.incoming.rparticles[2] += incoming_parameters[3]  # mu_y
        self.incoming.rparticles[3] += incoming_parameters[4]  # # mu_yp

        if self.misalignment_mode == "constant":
            self.misalignments = self.misalignment_values
        elif self.misalignment_mode == "random":
            self.misalignments = self.misalignment_space.sample()
        else:
            raise ValueError(
                f'Invalid value "{self.misalignment_mode}" for misalignment_mode'
            )
        self.lattice[areamqzm1].dx = self.misalignments[0]
        self.lattice[areamqzm1].dy = self.misalignments[1]
        self.lattice[areamqzm2].dx = self.misalignments[2]
        self.lattice[areamqzm2].dy = self.misalignments[3]
        self.lattice[areamqzm3].dx = self.misalignments[4]
        self.lattice[areamqzm3].dy = self.misalignments[5]
        self.screen_misalignment = self.misalignments[6:8]

    def update(self) -> None:
        self.outbeam = deepcopy(self.incoming)
        navi = oc.Navigator(self.lattice)
        if self.include_space_charge:
            navi.unit_step = self.unit_step
            navi.add_physics_proc(
                self.sc, self.lattice.sequence[0], self.lattice.sequence[-1]
            )
        _, self.outbeam = oc.track(
            self.lattice, self.outbeam, navi, print_progress=False
        )

    def get_beam_parameters(self) -> np.ndarray:
        mu_x = np.mean(self.outbeam.rparticles[0])
        sigma_x = np.std(self.outbeam.rparticles[0])
        mu_y = np.mean(self.outbeam.rparticles[2])
        sigma_y = np.std(self.outbeam.rparticles[2])

        # Apply screen misalignment
        mu_x -= self.screen_misalignment[0]
        mu_y -= self.screen_misalignment[1]

        return np.array([mu_x, sigma_x, mu_y, sigma_y])

    def get_incoming_parameters(self) -> np.ndarray:
        # Parameters of incoming are typed out to guarantee their order, as the
        # order would not be guaranteed creating np.array from dict.
        return np.array(self.incoming_parameters)

    def get_misalignments(self) -> np.ndarray:
        return np.array(self.misalignments)

    def get_binning(self) -> np.ndarray:
        return self.binning

    def get_screen_resolution(self) -> np.ndarray:
        return self.screen_resolution

    def get_pixel_size(self) -> np.ndarray:
        return self.screen_pixel_size


class EADOOCSBackend(TransverseTuningBaseBackend):
    """
    Backend for the ARES EA to communicate with the real accelerator through the DOOCS
    control system.
    """

    def __init__(self) -> None:
        self.beam_parameter_compute_failed = {"x": False, "y": False}
        self.reset_accelerator_was_just_called = False

    def is_beam_on_screen(self):
        return not all(self.beam_parameter_compute_failed.values())

    def get_magnets(self):
        return np.array(
            [
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.RBV")["data"],
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.RBV")["data"],
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.RBV")["data"]
                / 1000,
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.RBV")["data"],
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.RBV")["data"]
                / 1000,
            ]
        )

    def set_magnets(self, magnets):
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.SP", magnets[0])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.SP", magnets[1])
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.SP", magnets[2] * 1000
        )
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.SP", magnets[3])
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.SP", magnets[4] * 1000
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

    def reset(self):
        self.update()

        self.magnets_before_reset = self.get_magnets()
        self.screen_before_reset = self.get_screen_image()
        self.beam_before_reset = self.get_beam_parameters()

        # In order to record a screen image right after the accelerator was reset, this
        # flag is set so that we know to record the image the next time
        # `update_accelerator` is called.
        self.reset_accelerator_was_just_called = True

    def update(self):
        self.screen_image = self.capture_clean_screen_image()

        # Record the beam image just after reset (because there is no info on reset).
        # It will be included in `info` of the next step.
        if self.reset_accelerator_was_just_called:
            self.screen_after_reset = self.screen_image
            self.reset_accelerator_was_just_called = False

    def get_beam_parameters(self):
        img = self.get_screen_image()
        pixel_size = self.get_pixel_size()
        resolution = self.get_screen_resolution()

        parameters = {}
        for axis, direction in zip([0, 1], ["x", "y"]):
            projection = img.sum(axis=axis)
            minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
            filtered = uniform_filter1d(
                minfiltered, size=5, mode="nearest"
            )  # TODO rethink filters

            (half_values,) = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2

                # If (almost) all pixels are in FWHM, the beam might not be on screen
                self.beam_parameter_compute_failed[direction] = (
                    len(half_values) > 0.95 * resolution[axis]
                )
            else:
                fwhm_pixel = 42  # TODO figure out what to do with these
                center_pixel = 42

            parameters[f"mu_{direction}"] = (
                center_pixel - len(filtered) / 2
            ) * pixel_size[axis]
            parameters[f"sigma_{direction}"] = fwhm_pixel / 2.355 * pixel_size[axis]

        parameters["mu_y"] = -parameters["mu_y"]

        return np.array(
            [
                parameters["mu_x"],
                parameters["sigma_x"],
                parameters["mu_y"],
                parameters["sigma_y"],
            ]
        )

    def get_screen_image(self):
        return self.screen_image

    def get_binning(self):
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

    def get_screen_resolution(self):
        return np.array(
            [
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/WIDTH")["data"],
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/HEIGHT")["data"],
            ]
        )

    def get_pixel_size(self):
        return (
            np.array(
                [
                    abs(
                        pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/X.POLY_SCALE")[
                            "data"
                        ][2]
                    )
                    / 1000,
                    abs(
                        pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/Y.POLY_SCALE")[
                            "data"
                        ][2]
                    )
                    / 1000,
                ]
            )
            * self.get_binning()
        )

    def capture_clean_screen_image(self, average=5):
        """
        Capture a clean image of the beam from the screen using `average` images with
        beam on and `average` images of the background and then removing the background.

        Saves the image to a property of the object.
        """
        # Laser off
        self.set_cathode_laser(False)
        background_images = self.capture_interval(n=average, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        self.set_cathode_laser(True)
        screen_images = self.capture_interval(n=average, dt=0.1)
        median_beam = np.median(screen_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16 - 1)
        flipped = np.flipud(removed)

        return flipped.astype(np.uint16)

    def capture_interval(self, n, dt):
        """Capture `n` images from the screen and wait `dt` seconds in between them."""
        images = []
        for _ in range(n):
            images.append(self.capture_screen())
            time.sleep(dt)
        return np.array(images)

    def capture_screen(self):
        """Capture and image from the screen."""
        return pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ")["data"]

    def set_cathode_laser(self, setto: bool) -> None:
        """
        Sets the bool switch of the cathode laser event to `setto` and waits a second.
        """
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)

    def get_info(self) -> dict:
        # If magnets or the beam were recorded before reset, add them info on the first
        # step, so a generalised data recording wrapper captures them.
        info = {}

        # Screen image
        info["screen_image"] = self.get_screen_image()

        if hasattr(self, "magnets_before_reset"):
            info["magnets_before_reset"] = self.magnets_before_reset
            del self.magnets_before_reset
        if hasattr(self, "screen_before_reset"):
            info["screen_before_reset"] = self.screen_before_reset
            del self.screen_before_reset
        if hasattr(self, "beam_before_reset"):
            info["beam_before_reset"] = self.beam_before_reset
            del self.beam_before_reset

        if hasattr(self, "screen_after_reset"):
            info["screen_after_reset"] = self.screen_after_reset
            del self.screen_after_reset

        # Gain of camera for AREABSCR1
        info["camera_gain"] = pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/GAINRAW")[
            "data"
        ]

        # Steerers upstream of Experimental Area
        for steerer in ["ARLIMCHM1", "ARLIMCVM1", "ARLIMCHM2", "ARLIMCVM2"]:
            response = pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{steerer}/KICK_MRAD.RBV")
            info[steerer] = response["data"] / 1000

        # Gun solenoid
        info["gun_solenoid"] = pydoocs.read(
            "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/FIELD.RBV"
        )["data"]

        return info


class BCCheetahBackend(TransverseTuningBaseBackend):
    """
    Cheetah simulation backend to the end of the ARES Matching Region right before the
    Bunch Compressor.
    """

    def __init__(
        self,
        incoming_mode: str = "random",
        incoming_values: Optional[np.ndarray] = None,
        max_misalignment: float = 5e-4,
        misalignment_mode: str = "random",
        misalignment_values: Optional[np.ndarray] = None,
    ) -> None:
        self.incoming_mode = incoming_mode
        self.incoming_values = incoming_values
        self.max_misalignment = max_misalignment
        self.misalignment_mode = misalignment_mode
        self.misalignment_values = misalignment_values

        # Set up domain randomisation spaces
        self.incoming_beam_space = spaces.Box(
            low=np.array(
                [
                    80e6,
                    -1e-3,
                    -1e-4,
                    -1e-3,
                    -1e-4,
                    1e-5,
                    1e-6,
                    1e-5,
                    1e-6,
                    1e-6,
                    1e-4,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3],
                dtype=np.float32,
            ),
        )
        self.misalignment_space = spaces.Box(
            low=-self.max_misalignment, high=self.max_misalignment, shape=(8,)
        )

        # Create particle simulation
        ocelot_cell = [
            ares_oc.drift_armrbscr1,
            ares_oc.armrmqzm4,
            ares_oc.drift_armrmqzm4,
            ares_oc.armreolt1,
            ares_oc.ardgsolo1,
            ares_oc.drift_ardgsolo1,
            ares_oc.ardgeolo1,
            ares_oc.armrsolb1,
            ares_oc.drift_armrsolb1,
            ares_oc.armrbpmg3,
            ares_oc.drift_armrbpmg3,
            ares_oc.armrmqzm5,
            ares_oc.drift_armrmqzm5,
            ares_oc.armrmcvm5,
            ares_oc.drift_armrmcvm5,
            ares_oc.armrmchm5,
            ares_oc.drift_armrmchm2,
            ares_oc.armrbscr3,
            ares_oc.drift_armrbscr1,
            ares_oc.armrmqzm6,
            ares_oc.drift_armrmqzm6,
            ares_oc.armreolb1,
            ares_oc.arbcsolc,
            ares_oc.drift_arbcsolc,
            ares_oc.arbcmbhb1,
            ares_oc.drift_arbcmbhb1,
            ares_oc.arbcmbhb2,
            ares_oc.drift_arbcmbhb2,
            ares_oc.arbcbpml1,
            ares_oc.drift_arbcbpml1,
            ares_oc.arbcslhb1,
            ares_oc.drift_arbcslhb1,
            ares_oc.arbcslhs1,
            ares_oc.drift_arbcslhs1,
            ares_oc.arbcbsce1,
        ]
        self.segment = cheetah.Segment.from_ocelot(
            ocelot_cell, warnings=False, device="cpu"
        )  # .subcell("Drift_ARMRBSCR1", "ARBCBSCE1")
        self.segment.ARBCBSCE1.resolution = (2463, 2055)
        self.segment.ARBCBSCE1.pixel_size = (6.0000e-6, 4.1200e-6)
        self.segment.ARBCBSCE1.binning = 4
        self.segment.ARBCBSCE1.is_active = True

    def is_beam_on_screen(self) -> bool:
        screen = self.segment.ARBCBSCE1
        beam_position = np.array([screen.read_beam.mu_x, screen.read_beam.mu_y])
        limits = np.array(screen.resolution) / 2 * np.array(screen.pixel_size)
        return np.all(np.abs(beam_position) < limits)

    def get_magnets(self) -> np.ndarray:
        return np.array(
            [
                self.segment.ARMRMQZM4.k1,
                self.segment.ARMRMQZM5.k1,
                self.segment.ARMRMCVM5.angle,
                self.segment.ARMRMCHM5.angle,
                self.segment.ARMRMQZM6.k1,
            ]
        )

    def set_magnets(self, magnets: np.ndarray) -> None:
        self.segment.ARMRMQZM4.k1 = magnets[0]
        self.segment.ARMRMQZM5.k1 = magnets[1]
        self.segment.ARMRMCVM5.angle = magnets[2]
        self.segment.ARMRMCHM5.angle = magnets[3]
        self.segment.ARMRMQZM6.k1 = magnets[4]

    def reset(self) -> None:
        # New domain randomisation
        if self.incoming_mode == "constant":
            incoming_parameters = self.incoming_values
        elif self.incoming_mode == "random":
            incoming_parameters = self.incoming_beam_space.sample()
        else:
            raise ValueError(f'Invalid value "{self.incoming_mode}" for incoming_mode')
        self.incoming = cheetah.ParameterBeam.from_parameters(
            energy=incoming_parameters[0],
            mu_x=incoming_parameters[1],
            mu_xp=incoming_parameters[2],
            mu_y=incoming_parameters[3],
            mu_yp=incoming_parameters[4],
            sigma_x=incoming_parameters[5],
            sigma_xp=incoming_parameters[6],
            sigma_y=incoming_parameters[7],
            sigma_yp=incoming_parameters[8],
            sigma_s=incoming_parameters[9],
            sigma_p=incoming_parameters[10],
        )

        if self.misalignment_mode == "constant":
            misalignments = self.misalignment_values
        elif self.misalignment_mode == "random":
            misalignments = self.misalignment_space.sample()
        else:
            raise ValueError(
                f'Invalid value "{self.misalignment_mode}" for misalignment_mode'
            )
        self.segment.ARMRMQZM4.misalignment = misalignments[0:2]
        self.segment.ARMRMQZM5.misalignment = misalignments[2:4]
        self.segment.ARMRMQZM6.misalignment = misalignments[4:6]
        self.segment.ARBCBSCE1.misalignment = misalignments[6:8]

    def update(self) -> None:
        self.segment(self.incoming)

    def get_beam_parameters(self) -> np.ndarray:
        return np.array(
            [
                self.segment.ARBCBSCE1.read_beam.mu_x,
                self.segment.ARBCBSCE1.read_beam.sigma_x,
                self.segment.ARBCBSCE1.read_beam.mu_y,
                self.segment.ARBCBSCE1.read_beam.sigma_y,
            ]
        )

    def get_incoming_parameters(self) -> np.ndarray:
        # Parameters of incoming are typed out to guarantee their order, as the
        # order would not be guaranteed creating np.array from dict.
        return np.array(
            [
                self.incoming.energy,
                self.incoming.mu_x,
                self.incoming.mu_xp,
                self.incoming.mu_y,
                self.incoming.mu_yp,
                self.incoming.sigma_x,
                self.incoming.sigma_xp,
                self.incoming.sigma_y,
                self.incoming.sigma_yp,
                self.incoming.sigma_s,
                self.incoming.sigma_p,
            ]
        )

    def get_misalignments(self) -> np.ndarray:
        return np.array(
            [
                self.segment.ARMRMQZM4.misalignment[0],
                self.segment.ARMRMQZM4.misalignment[1],
                self.segment.ARMRMQZM5.misalignment[0],
                self.segment.ARMRMQZM5.misalignment[1],
                self.segment.ARMRMQZM6.misalignment[0],
                self.segment.ARMRMQZM6.misalignment[1],
                self.segment.ARBCBSCE1.misalignment[0],
                self.segment.ARBCBSCE1.misalignment[1],
            ],
            dtype=np.float32,
        )

    def get_screen_image(self) -> np.ndarray:
        # Screen image to look like real image by dividing by goodlooking number and
        # scaling to 12 bits)
        return self.segment.ARBCBSCE1.reading / 1e9 * 2**12

    def get_binning(self) -> np.ndarray:
        return np.array(self.segment.ARBCBSCE1.binning)

    def get_screen_resolution(self) -> np.ndarray:
        return np.array(self.segment.ARBCBSCE1.resolution) / self.get_binning()

    def get_pixel_size(self) -> np.ndarray:
        return np.array(self.segment.ARBCBSCE1.pixel_size) * self.get_binning()

    def get_info(self) -> dict:
        return {
            "incoming_beam": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
        }


class BCDOOCSBackend(TransverseTuningBaseBackend):
    """
    Backend for the ARES Matching Region right before the Bunch Compressor to
    communicate with the real accelerator through the DOOCS control system.
    """

    def __init__(self) -> None:
        self.beam_parameter_compute_failed = {"x": False, "y": False}
        self.reset_accelerator_was_just_called = False

    def is_beam_on_screen(self):
        return not all(self.beam_parameter_compute_failed.values())

    def get_magnets(self):
        return np.array(
            [
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/ARMRMQZM4/STRENGTH.RBV")["data"],
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/ARMRMQZM5/STRENGTH.RBV")["data"],
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/ARMRMCVM5/KICK_MRAD.RBV")["data"]
                / 1000,
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/ARMRMCHM5/KICK_MRAD.RBV")["data"]
                / 1000,
                pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/ARMRMQZM6/STRENGTH.RBV")["data"],
            ]
        )

    def set_magnets(self, magnets):
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/ARMRMQZM4/STRENGTH.SP", magnets[0])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/ARMRMQZM5/STRENGTH.SP", magnets[1])
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/ARMRMCVM5/KICK_MRAD.SP", magnets[2] * 1000
        )
        pydoocs.write(
            "SINBAD.MAGNETS/MAGNET.ML/ARMRMCHM5/KICK_MRAD.SP", magnets[3] * 1000
        )
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/ARMRMQZM6/STRENGTH.SP", magnets[4])

        # Wait until magnets have reached their setpoints

        time.sleep(3.0)  # Wait for magnets to realise they received a command

        magnets = ["ARMRMQZM4", "ARMRMQZM5", "ARMRMCVM5", "ARMRMCHM5", "ARMRMQZM6"]

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

    def reset(self):
        self.update()

        self.magnets_before_reset = self.get_magnets()
        self.screen_before_reset = self.get_screen_image()
        self.beam_before_reset = self.get_beam_parameters()

        # In order to record a screen image right after the accelerator was reset, this
        # flag is set so that we know to record the image the next time
        # `update_accelerator` is called.
        self.reset_accelerator_was_just_called = True

    def update(self):
        self.screen_image = self.capture_clean_screen_image()

        # Record the beam image just after reset (because there is no info on reset).
        # It will be included in `info` of the next step.
        if self.reset_accelerator_was_just_called:
            self.screen_after_reset = self.screen_image
            self.reset_accelerator_was_just_called = False

    def get_beam_parameters(self):
        img = self.get_screen_image()
        pixel_size = self.get_pixel_size()
        resolution = self.get_screen_resolution()

        parameters = {}
        for axis, direction in zip([0, 1], ["x", "y"]):
            projection = img.sum(axis=axis)
            minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
            filtered = uniform_filter1d(
                minfiltered, size=5, mode="nearest"
            )  # TODO rethink filters

            (half_values,) = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2

                # If (almost) all pixels are in FWHM, the beam might not be on screen
                self.beam_parameter_compute_failed[direction] = (
                    len(half_values) > 0.95 * resolution[axis]
                )
            else:
                fwhm_pixel = 42  # TODO figure out what to do with these
                center_pixel = 42

            parameters[f"mu_{direction}"] = (
                center_pixel - len(filtered) / 2
            ) * pixel_size[axis]
            parameters[f"sigma_{direction}"] = fwhm_pixel / 2.355 * pixel_size[axis]

        parameters["mu_y"] = -parameters["mu_y"]

        return np.array(
            [
                parameters["mu_x"],
                parameters["sigma_x"],
                parameters["mu_y"],
                parameters["sigma_y"],
            ]
        )

    def get_screen_image(self):
        return self.screen_image

    def get_binning(self):
        return np.array(
            (
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.BC.BSC.E.1/BINNINGHORIZONTAL")[
                    "data"
                ],
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.BC.BSC.E.1/BINNINGVERTICAL")[
                    "data"
                ],
            )
        )

    def get_screen_resolution(self):
        return np.array(
            [
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.BC.BSC.E.1/WIDTH")["data"],
                pydoocs.read("SINBAD.DIAG/CAMERA/AR.BC.BSC.E.1/HEIGHT")["data"],
            ]
        )

    def get_pixel_size(self):
        return (
            np.array(
                [
                    abs(
                        pydoocs.read("SINBAD.DIAG/CAMERA/AR.BC.BSC.E.1/X.POLY_SCALE")[
                            "data"
                        ][2]
                    )
                    / 1000,
                    abs(
                        pydoocs.read("SINBAD.DIAG/CAMERA/AR.BC.BSC.E.1/Y.POLY_SCALE")[
                            "data"
                        ][2]
                    )
                    / 1000,
                ]
            )
            * self.get_binning()
        )

    def capture_clean_screen_image(self, average=5):
        """
        Capture a clean image of the beam from the screen using `average` images with
        beam on and `average` images of the background and then removing the background.

        Saves the image to a property of the object.
        """
        # Laser off
        self.set_cathode_laser(False)
        background_images = self.capture_interval(n=average, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        self.set_cathode_laser(True)
        screen_images = self.capture_interval(n=average, dt=0.1)
        median_beam = np.median(screen_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16 - 1)
        flipped = np.flipud(removed)

        return flipped.astype(np.uint16)

    def capture_interval(self, n, dt):
        """Capture `n` images from the screen and wait `dt` seconds in between them."""
        images = []
        for _ in range(n):
            images.append(self.capture_screen())
            time.sleep(dt)
        return np.array(images)

    def capture_screen(self):
        """Capture and image from the screen."""
        return pydoocs.read("SINBAD.DIAG/CAMERA/AR.BC.BSC.E.1/IMAGE_EXT_ZMQ")["data"]

    def set_cathode_laser(self, setto: bool) -> None:
        """
        Sets the bool switch of the cathode laser event to `setto` and waits a second.
        """
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)

    def get_info(self) -> dict:
        # If magnets or the beam were recorded before reset, add them info on the first
        # step, so a generalised data recording wrapper captures them.
        info = {}

        # Screen image
        info["screen_image"] = self.get_screen_image()

        if hasattr(self, "magnets_before_reset"):
            info["magnets_before_reset"] = self.magnets_before_reset
            del self.magnets_before_reset
        if hasattr(self, "screen_before_reset"):
            info["screen_before_reset"] = self.screen_before_reset
            del self.screen_before_reset
        if hasattr(self, "beam_before_reset"):
            info["beam_before_reset"] = self.beam_before_reset
            del self.beam_before_reset

        if hasattr(self, "screen_after_reset"):
            info["screen_after_reset"] = self.screen_after_reset
            del self.screen_after_reset

        # Gain of camera for AREABSCR1
        info["camera_gain"] = pydoocs.read("SINBAD.DIAG/CAMERA/AR.BC.BSC.E.1/GAINRAW")[
            "data"
        ]

        # Steerers upstream of Experimental Area
        for steerer in ["ARMRMCHM3", "ARMRMCVM3", "ARMRMCVM4", "ARMRMCHM4"]:
            response = pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{steerer}/KICK_MRAD.RBV")
            info[steerer] = response["data"] / 1000

        # Gun solenoid
        info["gun_solenoid"] = pydoocs.read(
            "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/FIELD.RBV"
        )["data"]

        return info
