from copy import deepcopy
from typing import Optional

import numpy as np
from gym import spaces
from ocelot import (  # XYQuadrupole,
    MagneticLattice,
    MethodTM,
    Navigator,
    SecondTM,
    SpaceCharge,
    track,
)
from ocelot.cpbd.beam import generate_parray

from ARESlatticeStage3v1_9 import (
    areabscr1,
    areamchm1,
    areamcvm1,
    areamqzm1,
    areamqzm2,
    areamqzm3,
    areasola1,
    drift_areamchm1,
    drift_areamcvm1,
    drift_areamqzm1,
    drift_areamqzm2,
    drift_areamqzm3,
    drift_areasola1,
)
from ea_train import ARESEA

# To incorporate misalignments
# areamqzm1 = XYQuadrupole(l=0.122, eid="AREAMQZM1")
# areamqzm2 = XYQuadrupole(l=0.122, eid="AREAMQZM2")
# areamqzm3 = XYQuadrupole(l=0.122, eid="AREAMQZM3")


class ARESEAOcelot(ARESEA):
    def __init__(
        self,
        incoming_mode: str = "random",
        incoming_values: Optional[np.ndarray] = None,
        max_misalignment: float = 5e-4,
        misalignment_mode: str = "random",
        misalignment_values: Optional[np.ndarray] = None,
        action_mode: str = "direct",
        beam_distance_ord: int = 1,
        include_beam_image_in_info: bool = False,
        log_beam_distance: bool = False,
        magnet_init_mode: Optional[str] = None,
        magnet_init_values: Optional[np.ndarray] = None,
        max_quad_delta: Optional[float] = None,
        max_steerer_delta: Optional[float] = None,
        normalize_beam_distance: bool = True,
        reward_mode: str = "differential",
        target_beam_mode: str = "random",
        target_beam_values: Optional[np.ndarray] = None,
        target_mu_x_threshold: float = 3.3198e-6,
        target_mu_y_threshold: float = 2.4469e-6,
        target_sigma_x_threshold: float = 3.3198e-6,
        target_sigma_y_threshold: float = 2.4469e-6,
        threshold_hold: int = 1,
        w_beam: float = 1.0,
        w_done: float = 1.0,
        w_mu_x: float = 1.0,
        w_mu_x_in_threshold: float = 1.0,
        w_mu_y: float = 1.0,
        w_mu_y_in_threshold: float = 1.0,
        w_on_screen: float = 1.0,
        w_sigma_x: float = 1.0,
        w_sigma_x_in_threshold: float = 1.0,
        w_sigma_y: float = 1.0,
        w_sigma_y_in_threshold: float = 1.0,
        w_time: float = 1.0,
        # ocelot-only parameteres
        include_sc: bool = True,
        charge: float = 1e-12,  # in C
        nparticles: int = 1e5,
        unit_step: float = 0.01,  # tracking step in [m]
    ) -> None:
        self.incoming_mode = incoming_mode
        self.incoming_values = incoming_values
        self.max_misalignment = max_misalignment
        self.misalignment_mode = misalignment_mode
        self.misalignment_values = misalignment_values

        super().__init__(
            action_mode=action_mode,
            beam_distance_ord=beam_distance_ord,
            include_beam_image_in_info=include_beam_image_in_info,
            log_beam_distance=log_beam_distance,
            magnet_init_mode=magnet_init_mode,
            magnet_init_values=magnet_init_values,
            max_quad_delta=max_quad_delta,
            max_steerer_delta=max_steerer_delta,
            normalize_beam_distance=normalize_beam_distance,
            reward_mode=reward_mode,
            target_beam_mode=target_beam_mode,
            target_beam_values=target_beam_values,
            target_mu_x_threshold=target_mu_x_threshold,
            target_mu_y_threshold=target_mu_y_threshold,
            target_sigma_x_threshold=target_sigma_x_threshold,
            target_sigma_y_threshold=target_sigma_y_threshold,
            threshold_hold=threshold_hold,
            w_beam=w_beam,
            w_done=w_done,
            w_mu_x=w_mu_x,
            w_mu_x_in_threshold=w_mu_x_in_threshold,
            w_mu_y=w_mu_y,
            w_mu_y_in_threshold=w_mu_y_in_threshold,
            w_on_screen=w_on_screen,
            w_sigma_x=w_sigma_x,
            w_sigma_x_in_threshold=w_sigma_x_in_threshold,
            w_sigma_y=w_sigma_y,
            w_sigma_y_in_threshold=w_sigma_y_in_threshold,
            w_time=w_time,
        )
        self.include_sc = include_sc
        self.nparticles = int(nparticles)
        self.charge = charge

        # Initialize Tracking method
        self.method = MethodTM()
        self.method.global_method = SecondTM
        self.unit_step = unit_step

        self.sc = SpaceCharge()
        self.sc.nmesh_xyz = [32, 32, 32]
        self.sc.step = 1

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

        # Build lattice
        self.lattice = MagneticLattice(
            self.cell, start=areasola1, stop=areabscr1, method=self.method
        )

        # For consistency in logging, not actually used
        self.binning = 4
        self.screen_resolution = (2448, 2040)
        self.screen_pixel_size = (3.3198e-6, 2.4469e-6)

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

    def reset_accelerator(self) -> None:
        # New domain randomisation
        if self.incoming_mode == "constant":
            self.incoming_parameters = self.incoming_values
        elif self.incoming_mode == "random":
            self.incoming_parameters = self.observation_space["incoming"].sample()
        else:
            raise ValueError(f'Invalid value "{self.incoming_mode}" for incoming_mode')
        self.incoming = generate_parray(
            sigma_x=self.incoming_parameters[5],
            sigma_px=self.incoming_parameters[6],
            sigma_y=self.incoming_parameters[7],
            sigma_py=self.incoming_parameters[8],
            sigma_tau=self.incoming_parameters[9],
            sigma_p=self.incoming_parameters[10],
            energy=self.incoming_parameters[0] / 1e9,
            nparticles=self.nparticles,
            charge=self.charge,
        )
        # Set mu_x, mu_xp, mu_y, mu_yp
        self.incoming.rparticles[0] += self.incoming_parameters[1]  # mu_x
        self.incoming.rparticles[1] += self.incoming_parameters[2]  # mu_xp
        self.incoming.rparticles[2] += self.incoming_parameters[3]  # mu_y
        self.incoming.rparticles[3] += self.incoming_parameters[4]  # # mu_yp

        if self.misalignment_mode == "constant":
            self.misalignments = self.misalignment_values
        elif self.misalignment_mode == "random":
            self.misalignments = self.observation_space["misalignments"].sample()
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

    def update_accelerator(self) -> None:
        self.outbeam = deepcopy(self.incoming)
        navi = Navigator(self.lattice)
        if self.include_sc:
            navi.unit_step = self.unit_step
            navi.add_physics_proc(
                self.sc, self.lattice.sequence[0], self.lattice.sequence[-1]
            )
        _, self.outbeam = track(self.lattice, self.outbeam, navi, print_progress=False)

    def get_beam_parameters(self) -> np.ndarray:
        beam_parameters = np.mean(self.outbeam.rparticles, axis=1)[0:4]
        # Apply screen misalignment
        beam_parameters[0] -= self.screen_misalignment[0]
        beam_parameters[2] -= self.screen_misalignment[1]
        return beam_parameters

    def get_incoming_parameters(self) -> np.ndarray:
        # Parameters of incoming are typed out to guarantee their order, as the
        # order would not be guaranteed creating np.array from dict.
        return np.array(self.incoming_parameters)

    def get_misalignments(self) -> np.ndarray:
        return np.array(self.misalignments)

    # def get_beam_image(self) -> np.ndarray:
    #     # Beam image to look like real image by dividing by goodlooking number and
    #     # scaling to 12 bits)
    #     return self.simulation.AREABSCR1.reading / 1e9 * 2**12

    def get_binning(self) -> np.ndarray:
        return np.array(self.binning)

    def get_screen_resolution(self) -> np.ndarray:
        # Not making sense here
        return np.array(self.screen_resolution) / self.get_binning()

    def get_pixel_size(self) -> np.ndarray:
        # Not making sense here
        return np.array(self.screen_pixel_size) * self.get_binning()

    def get_accelerator_observation_space(self) -> dict:
        return {
            "incoming": spaces.Box(
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
            ),
            "misalignments": spaces.Box(
                low=-self.max_misalignment, high=self.max_misalignment, shape=(8,)
            ),
        }

    def get_accelerator_observation(self) -> dict:
        return {
            "incoming": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
        }
