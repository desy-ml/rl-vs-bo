import cheetah
import gym
from gym import spaces
import numpy as np

from ARESlatticeStage3v1_9 import cell as ares_lattice


class ARESEA(gym.Env):

    observation_space = spaces.Dict({
        "beam": spaces.Box(
            low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        ),
        "magnets": spaces.Box(
            low=np.array([0, 0, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
            high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
        )
    })

    # action_space= spaces.Box(
    #     low=np.array([0, 0, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
    #     high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
    # )
    action_space= spaces.Box(
        low=np.array([0], dtype=np.float32),
        high=np.array([72], dtype=np.float32)
    )

    def __init__(self):
        self.simulation = cheetah.Segment.from_ocelot(
            ares_lattice,
            warnings=False,
            device="cpu"
        ).subcell("AREASOLA1", "AREABSCR1")
        self.simulation.AREABSCR1.is_active = True

        misalignment_space = spaces.Box(low=-400e-6, high=400e-6, shape=(8,))
        misalignments = misalignment_space.sample()
        self.simulation.AREAMQZM1.misalignment = misalignments[0:2]
        self.simulation.AREAMQZM2.misalignment = misalignments[2:4]
        self.simulation.AREAMQZM3.misalignment = misalignments[4:6]
        self.simulation.AREABSCR1.misalignment = misalignments[6:8]

        self.incoming = cheetah.ParameterBeam.from_parameters(
            mu_x=np.random.uniform(-1e-3, 1e-3),
            mu_y=np.random.uniform(-1e-3, 1e-3),
            mu_xp=np.random.uniform(-1e-4, 1e-4),
            mu_yp=np.random.uniform(-1e-4, 1e-4),
            sigma_x=np.random.uniform(1e-5, 5e-4),
            sigma_y=np.random.uniform(1e-5, 5e-4),
            sigma_xp=np.random.uniform(1e-6, 5e-5),
            sigma_yp=np.random.uniform(1e-6, 5e-5),
            sigma_s=np.random.uniform(1e-6, 5e-5),
            sigma_p=np.random.uniform(1e-4, 1e-3),
            energy=np.random.uniform(80e6, 160e6)
        )
    
    def reset(self):
        self.simulation.AREAMQZM1.k1 = 0.0
        self.simulation.AREAMQZM2.k1 = 0.0
        self.simulation.AREAMCVM1.angle = 0.0
        self.simulation.AREAMQZM3.k1 = 0.0
        self.simulation.AREAMCHM1.angle = 0.0

        # Run the simulation
        self.simulation(self.incoming)

        observation = {
            "beam": np.array([
                self.simulation.AREABSCR1.read_beam.mu_x,
                self.simulation.AREABSCR1.read_beam.sigma_x,
                self.simulation.AREABSCR1.read_beam.mu_y,
                self.simulation.AREABSCR1.read_beam.sigma_y
            ]),
            "magnets": np.array([
                self.simulation.AREAMQZM1.k1,
                -self.simulation.AREAMQZM2.k1,  # NOTE the sign here
                self.simulation.AREAMCVM1.angle,
                self.simulation.AREAMQZM3.k1,
                self.simulation.AREAMCHM1.angle
            ], dtype=np.float32)
        }

        self.previous_mae = observation["beam"].mean()

        return observation

    def step(self, action):
        self.simulation.AREAMQZM1.k1 = action[0]
        # self.simulation.AREAMQZM2.k1 = -action[1]  # NOTE the sign here
        # self.simulation.AREAMCVM1.angle = action[2]
        # self.simulation.AREAMQZM3.k1 = action[3]
        # self.simulation.AREAMCHM1.angle = action[4]

        # Run the simulation
        self.simulation(self.incoming)

        observation = {
            "beam": np.array([
                self.simulation.AREABSCR1.read_beam.mu_x,
                self.simulation.AREABSCR1.read_beam.sigma_x,
                self.simulation.AREABSCR1.read_beam.mu_y,
                self.simulation.AREABSCR1.read_beam.sigma_y
            ]),
            "magnets": np.array([
                self.simulation.AREAMQZM1.k1,
                -self.simulation.AREAMQZM2.k1,  # NOTE the sign here
                self.simulation.AREAMCVM1.angle,
                self.simulation.AREAMQZM3.k1,
                self.simulation.AREAMCHM1.angle
            ])
        }

        time_reward = -1
        mae = observation["beam"].mean()
        mae_reward = self.previous_mae - mae
        self.previous_mae = mae
        reward = 1 * time_reward + 1 * mae_reward

        done = bool(np.all(observation["beam"] < 3.3198e-6))

        info = {
            "mae": None,
            "misalignments": None,
            "incoming": None
        }

        return observation, reward, done, info
