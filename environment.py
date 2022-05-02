import cheetah
import cv2
import gym
from gym import spaces
import numpy as np

from ARESlatticeStage3v1_9 import cell as ares_lattice


class ARESEA(gym.Env):

    metadata = {
        "render.modes": ["rgb_array"],
        "video.frames_per_second": 2
    }

    observation_space = spaces.Dict({
        "beam": spaces.Box(
            low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        ),
        "magnets": spaces.Box(
            low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
            high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
        ),
        "incoming": spaces.Box(
            low=np.array([80e6, -1e-3, -1e-4, -1e-3, -1e-4, 1e-5, 1e-6, 1e-5, 1e-6, 1e-6, 1e-4], dtype=np.float32),
            high=np.array([160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3], dtype=np.float32)
        ),
        "misalignments": spaces.Box(low=-400e-6, high=400e-6, shape=(8,))
    })

    action_space = spaces.Box(
        low=np.array([0, 0, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
        high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
    )

    def __init__(self, misalignments=None, incoming_parameters=None):
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

        # if misalignments is None:
        #     misalignments = self.observation_space["misalignments"].sample()
        # self.simulation.AREAMQZM1.misalignment = misalignments[0:2]
        # self.simulation.AREAMQZM2.misalignment = misalignments[2:4]
        # self.simulation.AREAMQZM3.misalignment = misalignments[4:6]
        # self.simulation.AREABSCR1.misalignment = misalignments[6:8]

        # if incoming_parameters is None:
        #     incoming_parameters = self.observation_space["incoming"].sample()
        # self.incoming = cheetah.ParameterBeam.from_parameters(
        #     energy=incoming_parameters[0],
        #     mu_x=incoming_parameters[1],
        #     mu_xp=incoming_parameters[2],
        #     mu_y=incoming_parameters[3],
        #     mu_yp=incoming_parameters[4],
        #     sigma_x=incoming_parameters[5],
        #     sigma_xp=incoming_parameters[6],
        #     sigma_y=incoming_parameters[7],
        #     sigma_yp=incoming_parameters[8],
        #     sigma_s=incoming_parameters[9],
        #     sigma_p=incoming_parameters[10],
        # )
    
    def reset(self):
        self.simulation.AREAMQZM1.k1 = 0.0
        self.simulation.AREAMQZM2.k1 = -0.0     # NOTE the sign here
        self.simulation.AREAMCVM1.angle = 0.0
        self.simulation.AREAMQZM3.k1 = 0.0
        self.simulation.AREAMCHM1.angle = 0.0

        # New random setup
        misalignments = self.observation_space["misalignments"].sample()
        self.simulation.AREAMQZM1.misalignment = misalignments[0:2]
        self.simulation.AREAMQZM2.misalignment = misalignments[2:4]
        self.simulation.AREAMQZM3.misalignment = misalignments[4:6]
        self.simulation.AREABSCR1.misalignment = misalignments[6:8]

        incoming_parameters = self.observation_space["incoming"].sample()
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

        # Run the simulation
        self.simulation(self.incoming)

        self.initial_screen_beam = self.simulation.AREABSCR1.read_beam

        observation = {
            "beam": np.array([
                self.simulation.AREABSCR1.read_beam.mu_x,
                self.simulation.AREABSCR1.read_beam.sigma_x,
                self.simulation.AREABSCR1.read_beam.mu_y,
                self.simulation.AREABSCR1.read_beam.sigma_y
            ], dtype=np.float32),
            "magnets": np.array([
                self.simulation.AREAMQZM1.k1,
                -self.simulation.AREAMQZM2.k1,  # NOTE the sign here
                self.simulation.AREAMCVM1.angle,
                self.simulation.AREAMQZM3.k1,
                self.simulation.AREAMCHM1.angle
            ], dtype=np.float32),
            "incoming": np.array([
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
                self.incoming.sigma_p
            ], dtype=np.float32),
            "misalignments": np.array([
                self.simulation.AREAMQZM1.misalignment[0],
                self.simulation.AREAMQZM1.misalignment[1],
                self.simulation.AREAMQZM2.misalignment[0],
                self.simulation.AREAMQZM2.misalignment[1],
                self.simulation.AREAMQZM3.misalignment[0],
                self.simulation.AREAMQZM3.misalignment[1],
                self.simulation.AREABSCR1.misalignment[0],
                self.simulation.AREABSCR1.misalignment[1],
            ], dtype=np.float32)
        }

        return observation

    def step(self, action):
        # Get beam parameters before action is performed
        previous_beam = self.simulation.AREABSCR1.read_beam

        # Perform action
        self.simulation.AREAMQZM1.k1 = action[0]
        self.simulation.AREAMQZM2.k1 = -action[1]  # NOTE the sign here
        self.simulation.AREAMCVM1.angle = action[2]
        self.simulation.AREAMQZM3.k1 = action[3]
        self.simulation.AREAMCHM1.angle = action[4]

        # Run the simulation
        self.simulation(self.incoming)

        # Build observation
        observation = {
            "beam": np.array([
                self.simulation.AREABSCR1.read_beam.mu_x,
                self.simulation.AREABSCR1.read_beam.sigma_x,
                self.simulation.AREABSCR1.read_beam.mu_y,
                self.simulation.AREABSCR1.read_beam.sigma_y
            ], dtype=np.float32),
            "magnets": np.array([
                self.simulation.AREAMQZM1.k1,
                -self.simulation.AREAMQZM2.k1,  # NOTE the sign here
                self.simulation.AREAMCVM1.angle,
                self.simulation.AREAMQZM3.k1,
                self.simulation.AREAMCHM1.angle
            ], dtype=np.float32),
            "incoming": np.array([
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
                self.incoming.sigma_p
            ], dtype=np.float32),
            "misalignments": np.array([
                self.simulation.AREAMQZM1.misalignment[0],
                self.simulation.AREAMQZM1.misalignment[1],
                self.simulation.AREAMQZM2.misalignment[0],
                self.simulation.AREAMQZM2.misalignment[1],
                self.simulation.AREAMQZM3.misalignment[0],
                self.simulation.AREAMQZM3.misalignment[1],
                self.simulation.AREABSCR1.misalignment[0],
                self.simulation.AREABSCR1.misalignment[1],
            ], dtype=np.float32)
        }

        # Compute reward
        current_beam = self.simulation.AREABSCR1.read_beam

        time_reward = -1
        on_screen_reward = -(not self.is_beam_on_screen())
        mu_x_reward = (abs(float(previous_beam.mu_x) - float(self.initial_screen_beam.mu_x)) - abs(float(current_beam.mu_x) - float(self.initial_screen_beam.mu_x))) / abs(float(self.initial_screen_beam.mu_x))
        sigma_x_reward = float(previous_beam.sigma_x - current_beam.sigma_x) / abs(float(self.initial_screen_beam.sigma_x))
        mu_y_reward = (abs(float(previous_beam.mu_y) - float(self.initial_screen_beam.mu_y)) - abs(float(current_beam.mu_y) - float(self.initial_screen_beam.mu_y))) / abs(float(self.initial_screen_beam.mu_y))
        sigma_y_reward = float(previous_beam.sigma_y - current_beam.sigma_y) / abs(float(self.initial_screen_beam.sigma_y))

        # TODO: Maybe add aspect ratio term
        reward = 1 * time_reward + 0 * on_screen_reward + 0 * mu_x_reward + 1 * sigma_x_reward + 0 * mu_y_reward + 1 * sigma_y_reward

        # Figure out if reach good enough beam (done)
        done = bool(np.all(observation["beam"] < 3.3198e-6))

        # Put together info
        misalignments = {
            "AREAMQZM1": self.simulation.AREAMQZM1.misalignment,
            "AREAMQZM2": self.simulation.AREAMQZM2.misalignment,
            "AREAMQZM3": self.simulation.AREAMQZM3.misalignment,
            "AREABSCR1": self.simulation.AREABSCR1.misalignment,
        }
        info = {
            "misalignments": misalignments,
            "incoming": self.incoming.parameters,
            "time_reward": 1 * time_reward,
            "on_screen_reward": 0 * on_screen_reward,
            "mu_x_reward": 0 * mu_x_reward,
            "sigma_x_reward": 1 * sigma_x_reward,
            "mu_y_reward": 0 * mu_y_reward,
            "sigma_y_reward": 1 * sigma_y_reward
        }

        return observation, reward, done, info
    
    def render(self, mode="human"):
        assert mode == "rgb_array" or mode == "human"

        # Read screen image and make 8-bit RGB
        img = self.simulation.AREABSCR1.reading
        img = img / 1e9 * 255
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

    def is_beam_on_screen(self):
        screen = self.simulation.AREABSCR1
        beam_position = np.array([screen.read_beam.mu_x, screen.read_beam.mu_y])
        limits = np.array(screen.resolution) / 2 * np.array(screen.pixel_size)
        return np.all(np.abs(beam_position) < limits)
