import cheetah
import gym
from gym import spaces
import numpy as np

from .ares import ARESlatticeStage3v1_9 as lattice


class ARESEAMisalignments(gym.Env):

    screen_resolution = np.array([2448, 2040])
    pixel_size = np.array([3.3198e-6, 2.4469e-6])

    observation_space = spaces.Box(
        low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3, -np.inf, -np.inf, 0, 0], dtype=np.float32),
        high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
    )

    action_space = spaces.Box(
        low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3, -400e-6, -400e-6, -400e-6, -400e-6, -400e-6, -400e-6], dtype=np.float32),
        high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3, 400e-6, 400e-6, 400e-6, 400e-6, 400e-6, 400e-6], dtype=np.float32)
    )

    def __init__(self):
        self.segment = cheetah.Segment.from_ocelot(lattice.cell, warnings=False, device="cpu").subcell("AREASOLA1", "AREABSCR1")
        self.segment.AREABSCR1.resolution = self.screen_resolution
        self.segment.AREABSCR1.pixel_size = self.pixel_size
        # self.segment.AREABSCR1.is_active = True
        self.segment.AREABSCR1.binning = 4
    
    def reset(self):
        # Randomise true misalignments and incoming beam
        self.segment.AREAMQZM1.misalignment = np.random.uniform(-400e-6, 400e-6, size=2)
        self.segment.AREAMQZM2.misalignment = np.random.uniform(-400e-6, 400e-6, size=2)
        self.segment.AREAMQZM3.misalignment = np.random.uniform(-400e-6, 400e-6, size=2)
        self.segment.AREABSCR1.misalignment = np.random.uniform(-400e-6, 400e-6, size=2)

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

        # Compute relative misalignmnets and incoming beam under the assumption that when all
        # magnets are off, the beam travels on the 0-orbit. This means that all the beam's mus are 0.
        # We can also compute the screen's misalignment from the beam's position on the screen,
        # meaning that the agent only needs to predict the quadrupole misalignments.
        D_CENTER_AREAMQZM1 = 0.17504 + 0.122 / 2
        position_at_areamqzm1 = np.array([
            self.incoming.mu_x + self.incoming.mu_xp * D_CENTER_AREAMQZM1,
            self.incoming.mu_y + self.incoming.mu_yp * D_CENTER_AREAMQZM1
        ])
        relative_misalignment_areamqzm1 = np.array(self.segment.AREAMQZM1.misalignment) - position_at_areamqzm1

        D_CENTER_AREAMQZM2 = 0.17504 + 0.122 + 0.428 + 0.122 / 2
        position_at_areamqzm2 = np.array([
            self.incoming.mu_x + self.incoming.mu_xp * D_CENTER_AREAMQZM2,
            self.incoming.mu_y + self.incoming.mu_yp * D_CENTER_AREAMQZM2
        ])
        relative_misalignment_areamqzm2 = np.array(self.segment.AREAMQZM2.misalignment) - position_at_areamqzm2

        D_CENTER_AREAMQZM3 = 0.17504 + 0.122 + 0.428 + 0.122 + 0.204 + 0.02 + 0.204 + 0.122 / 2
        position_at_areamqzm3 = np.array([
            self.incoming.mu_x + self.incoming.mu_xp * D_CENTER_AREAMQZM3,
            self.incoming.mu_y + self.incoming.mu_yp * D_CENTER_AREAMQZM3
        ])
        relative_misalignment_areamqzm3 = np.array(self.segment.AREAMQZM3.misalignment) - position_at_areamqzm3

        self.relative_misalignments = np.stack([
            relative_misalignment_areamqzm1,
            relative_misalignment_areamqzm2,
            relative_misalignment_areamqzm3
        ])

        # Set magnets to zero
        self.segment.AREAMQZM1.k1 = 0
        self.segment.AREAMQZM2.k1 = 0
        self.segment.AREAMCVM1.angle = 0
        self.segment.AREAMQZM3.k1 = 0
        self.segment.AREAMCHM1.angle = 0

        # Construct observation
        outgoing = self.segment(self.incoming)
        observation = np.array([
            self.segment.AREAMQZM1.k1,
            self.segment.AREAMQZM2.k1,
            self.segment.AREAMCVM1.angle,
            self.segment.AREAMQZM3.k1,
            self.segment.AREAMCHM1.angle,
            outgoing.mu_x,
            outgoing.mu_y,
            outgoing.sigma_x,
            outgoing.sigma_y
        ])

        return observation
    
    def step(self, action):
        # Set magnets to new setting
        self.segment.AREAMQZM1.k1 = action[0]
        self.segment.AREAMQZM2.k1 = action[1]
        self.segment.AREAMCVM1.angle = action[2]
        self.segment.AREAMQZM3.k1 = action[3]
        self.segment.AREAMCHM1.angle = action[4]

        # Compute reward
        estimated_misalignments = action[5:].reshape((3,2))
        shaped_reward = -np.abs(self.relative_misalignments - estimated_misalignments).mean()
        prize_reward = np.sum((np.abs(self.relative_misalignments - estimated_misalignments) < 1e-6) * 100)
        reward = shaped_reward + prize_reward

         # Construct observation
        outgoing = self.segment(self.incoming)
        observation = np.array([
            self.segment.AREAMQZM1.k1,
            self.segment.AREAMQZM2.k1,
            self.segment.AREAMCVM1.angle,
            self.segment.AREAMQZM3.k1,
            self.segment.AREAMCHM1.angle,
            outgoing.mu_x,
            outgoing.mu_y,
            outgoing.sigma_x,
            outgoing.sigma_y
        ])

        # Check if done
        done = bool(np.all(np.abs(self.relative_misalignments - estimated_misalignments) < 1e-6))

        # Put together info
        info = {"true_relative_misalignments": self.relative_misalignments}

        return observation, reward, done, info
