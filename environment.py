import cheetah
import cv2
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
        self.simulation.AREABSCR1.resolution = (2448, 2040)
        self.simulation.AREABSCR1.pixel_size = (3.3198e-6, 2.4469e-6)
        self.simulation.AREABSCR1.is_active = True
        self.simulation.AREABSCR1.binning = 4
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

        self.previous_mae = np.mean(np.abs(observation["beam"]))

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
            ], dtype=np.float32)
        }

        time_reward = -1
        mae = np.mean(np.abs(observation["beam"]))
        mae_reward = self.previous_mae - mae
        self.previous_mae = mae
        reward = 1 * time_reward + 1 * mae_reward

        done = bool(np.all(observation["beam"] < 3.3198e-6))

        misalignments = {
            "AREAMQZM1": self.simulation.AREAMQZM1.misalignment,
            "AREAMQZM2": self.simulation.AREAMQZM2.misalignment,
            "AREAMQZM3": self.simulation.AREAMQZM3.misalignment,
            "AREABSCR1": self.simulation.AREABSCR1.misalignment,
        }
        info = {
            "mae": mae,
            "misalignments": misalignments,
            "incoming": self.incoming.parameters,
            "mae_reward": mae_reward,
            "time_reward": time_reward
        }

        return observation, reward, done, info
    
    def render(self, mode="human"):
        assert mode == "rgb_array" or mode == "human"

        # Read screen image and make 8-bit RGB
        img = self.simulation.AREABSCR1.reading
        img = img / 1e7 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:,:,np.newaxis], 3, axis=-1)

        # Draw beam ellipse
        screen = self.simulation.AREABSCR1
        beam = screen.read_beam
        pixel_size = np.array(screen.pixel_size) * screen.binning
        resolution = np.array(screen.resolution) / screen.binning
        e_pos_x = int(beam.mu_x / pixel_size[0] + resolution[0] / 2)
        e_width_x = int(beam.sigma_x / pixel_size[0])
        e_pos_y = int(-beam.mu_y / pixel_size[1] + resolution[1] / 2)
        e_width_y = int(beam.sigma_y / pixel_size[1])
        red = (0, 0, 255)
        img = cv2.ellipse(img, (e_pos_x,e_pos_y), (e_width_x,e_width_y), 0, 0, 360, red, 2)
        
        # Adjust aspect ration
        new_width = int(img.shape[1] * pixel_size[0] / pixel_size[1])
        img = cv2.resize(img, (new_width,img.shape[0]))

        # Add magnet values
        padding = np.full((int(img.shape[0]*0.18),img.shape[1],3), fill_value=255, dtype=np.uint8)
        img = np.vstack([img, padding])
        black = (0, 0, 0)
        img = cv2.putText(img, f"Q1={self.simulation.AREAMQZM1.k1:.2f}", (15,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"Q2={self.simulation.AREAMQZM2.k1:.2f}", (215,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"CV={self.simulation.AREAMCVM1.angle*1e3:.2f}", (415,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"Q3={self.simulation.AREAMQZM3.k1:.2f}", (615,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"CH={self.simulation.AREAMCHM1.angle*1e3:.2f}", (15,585), cv2.FONT_HERSHEY_SIMPLEX, 1, black)

        if mode == "human":
            cv2.imshow("ARES EA", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
