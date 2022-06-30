from functools import partial
import time

import cheetah
import cv2
import gym
from gym import spaces
from gym.wrappers import FilterObservation, FlattenObservation, FrameStack, RecordVideo, RescaleAction, TimeLimit
import numpy as np
import yaml
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from ARESlatticeStage3v1_9 import cell as ares_lattice
from utils import FilterAction, PolishedDonkeyCompatibility


def main():
    config = {
        "action_type": "direct",    # Choose "direct" and "delta"
        "gamma": 0.99,
        "filter_action": None,  # None for default action or list of indicies
        "filter_observation": None, # Set to None or list of observation names to keep
        "frame_stack": None,    # None or number of frames
        "incoming": "random",   # "random" or parameters (`ARESEACheetah().observation_space["incoming"].sample()`)
        "magnet_init": "random",    # "random" or settings
        "misalignments": "random",  # "random" or misalignments (`ARESEACheetah().observation_space["misalignments"].sample()`)
        "n_envs": 1,
        "normalize_observation": True,
        "normalize_reward": True,
        "rescale_action": (-1, 1),
        "reward_method": "feedback",    # "differential" or "feedback"
        "sb3_device": "auto",
        "target_beam_setting": "random",    # "random" or target
        "target_beam_threshold": 3.3198e-6,
        "time_limit": 50,
        "total_timesteps": 800000,
        "quad_action": "symmetric", # "symmetric" or "oneway"
        "vec_env": "dummy", # "dummy" or "subproc"
        "w_mu_x": 1.0,
        "w_mu_y": 1.0,
        "w_on_screen": 1.0,
        "w_sigma_x": 1.0,
        "w_sigma_y": 1.0,
        "w_time": 1.0,
    }

    train(config)


def train(config):
    # Setup wandb
    wandb.init(
        project="ares-ea-v2",
        entity="msk-ipc",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config
    )
    config["wandb_run_name"] = wandb.run.name

    # Setup environments
    if config["vec_env"] == "dummy":
        env = DummyVecEnv([partial(make_env, config) for _ in range(config["n_envs"])])
    elif config["vec_env"] == "subproc":
        env = SubprocVecEnv([partial(make_env, config) for _ in range(config["n_envs"])])
    else:
        raise ValueError(f"Invalid value \"{config['vec_env']}\" for dummy")
    eval_env = DummyVecEnv([partial(make_env, config, record_video=True)])

    if config["normalize_observation"] or config["normalize_reward"]:
        env = VecNormalize(
            env,
            norm_obs=config["normalize_observation"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=config["normalize_observation"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
            training=False,
        )

    # Train
    model = PPO(
        "MlpPolicy",
        env,
        device=config["sb3_device"],
        gamma=config["gamma"],
        tensorboard_log=f"log/{config['wandb_run_name']}",
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        eval_env=eval_env,
        eval_freq=4000,
        callback=WandbCallback()
    )

    model.save(f"models/{wandb.run.name}/model")
    if config["normalize_observation"] or config["normalize_reward"]:
        env.save(f"models/{wandb.run.name}/normalizer")
    save_to_yaml(config, f"models/{wandb.run.name}/config")


def optimize(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    threshold=3.3198e-6,
    max_steps=50,
    model_name="polished-donkey-996",
    logbook=False,
    callback=None,
):
    """
    Function used for optimisation during operation.

    Note: Current version only works for polished-donkey-996.
    """
    # config = read_from_yaml(f"models/{model}/config")
    assert model_name == "polished-donkey-996", "Current version only works for polished-donkey-996."
    
    # Load the model
    model = TD3.load(f"models/{model_name}/model")
    
    # Create the environment
    def make_env_polished():
        env = ARESEADOOCS(
            dummy=False,
            action_mode="delta",
            magnet_init_mode="constant",
            magnet_init_values=np.array([10, -10, 0, 10, 0]),
            reward_mode="differential",
            target_beam_mode="constant",
            target_beam_values=np.array([target_mu_x, target_sigma_x, target_mu_y, target_sigma_y]),
            target_beam_threshold=3.3198e-6,
        )
        if max_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_steps)
        env = FlattenObservation(env)
        env = PolishedDonkeyCompatibility(env)
        env = RescaleAction(env, -1, 1)
        env = RecordVideo(env, "recordings_function_test")
        
        return env

    env = DummyVecEnv([make_env_polished])
    env = VecNormalize.load(f"models/{model_name}/vec_normalize.pkl", env)
    env.training = False

    # Actual optimisation
    observation = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
    env.close()


def make_env(config, record_video=False):
    env = ARESEACheetah(
        action_type=config["action_type"],
        incoming=config["incoming"],
        magnet_init=config["magnet_init"],
        misalignments=config["misalignments"],
        reward_method=config["reward_method"],
        quad_action=config["quad_action"],
        target_beam_setting=config["target_beam_setting"],
        target_beam_threshold=config["target_beam_threshold"],
        w_mu_x=config["w_mu_x"],
        w_mu_y=config["w_mu_y"],
        w_on_screen=config["w_on_screen"],
        w_sigma_x=config["w_sigma_x"],
        w_sigma_y=config["w_sigma_y"],
        w_time=config["w_time"],
    )
    if config["filter_observation"] is not None:
        env = FilterObservation(env, config["filter_observation"])
    if config["filter_action"] is not None:
        env = FilterAction(env, config["filter_action"], replace=0)
    if config["time_limit"] is not None:
        env = TimeLimit(env, config["time_limit"])
    env = FlattenObservation(env)
    if config["frame_stack"] is not None:
        env = FrameStack(env, config["frame_stack"])
    if config["rescale_action"] is not None:
        env = RescaleAction(env, config["rescale_action"][0], config["rescale_action"][1])
    env = Monitor(env)
    if record_video:
        env = RecordVideo(env, video_folder=f"recordings/{config['wandb_run_name']}")
    return env


class ARESEA(gym.Env):
    """
    Base class for beam positioning and focusing on AREABSCR1 in the ARES EA.

    Parameters
    ----------
    action_mode : str
        How actions work. Choose `"direct"`, `"direct_unidirectional_quads"` or `"delta"`.
    magnet_init_mode : str
        Magnet initialisation on `reset`. Set to `None`, `"random"` or `"constant"`. The
        `"constant"` setting requires `magnet_init_values` to be set.
    magnet_init_values : np.ndarray
        Values to set magnets to on `reset`. May only be set when `magnet_init_mode` is set to 
        `"constant"`.
    reward_mode : str
        How to compute the reward. Choose from `"feedback"` or `"differential"`.
    target_beam_mode : str
        Setting of target beam on `reset`. Choose from `"constant"` or `"random"`. The `"constant"`
        setting requires `target_beam_values` to be set.
    """

    metadata = {
        "render.modes": ["rgb_array"],
        "video.frames_per_second": 2
    }

    def __init__(
        self,
        action_mode="direct",
        magnet_init_mode=None,
        magnet_init_values=None,
        reward_mode="differential",
        target_beam_mode="random",
        target_beam_values=None,
        target_beam_threshold=3.3198e-6,
        w_mu_x=1.0,
        w_mu_y=1.0,
        w_on_screen=1.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
        w_time=1.0
    ):
        self.action_mode = action_mode
        self.magnet_init_mode = magnet_init_mode
        self.magnet_init_values = magnet_init_values
        self.reward_mode = reward_mode
        self.target_beam_mode = target_beam_mode
        self.target_beam_values = target_beam_values
        self.target_beam_threshold = target_beam_threshold
        self.w_mu_x = w_mu_x
        self.w_mu_y = w_mu_y
        self.w_on_screen = w_on_screen
        self.w_sigma_x = w_sigma_x
        self.w_sigma_y = w_sigma_y
        self.w_time = w_time

        # Create action space
        if self.action_mode == "direct":
            self.action_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
            )
        elif self.action_mode == "direct_unidirectional_quads":
            self.action_space = spaces.Box(
                low=np.array([0, -72, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 0, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
            )
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32) * 0.1,
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32) * 0.1
            )

        # Create observation space
        obs_space_dict = {
            "beam": spaces.Box(
                low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
            ),
            "magnets": self.action_space if self.action_mode.startswith("direct") else spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            ),
            "target": spaces.Box(
                low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
            )
        }
        obs_space_dict.update(self.get_accelerator_observation_space())
        self.observation_space = spaces.Dict(obs_space_dict)

        # Setup the accelerator (either simulation or the actual machine)
        self.setup_accelerator()
    
    def reset(self):
        self.reset_accelerator()

        if self.magnet_init_mode == "constant":
            self.set_magnets(self.magnet_init_values)
        elif self.magnet_init_mode == "random":
            self.set_magnets(self.observation_space["magnets"].sample())

        if self.target_beam_mode == "constant":
            self.target_beam = self.target_beam_values
        elif self.target_beam_mode == "random":
            self.target_beam = self.observation_space["target"].sample()

        # Update anything in the accelerator (mainly for running simulations)
        self.update_accelerator()

        self.initial_screen_beam = self.get_beam_parameters()
        self.previous_beam = self.initial_screen_beam

        observation = {
            "beam": self.initial_screen_beam.astype("float32"),
            "magnets": self.get_magnets().astype("float32"),
            "target": self.target_beam.astype("float32")
        }
        observation.update(self.get_accelerator_observation())

        return observation

    def step(self, action):
        # Perform action
        if self.action_mode.startswith("direct"):
            self.set_magnets(action)
        elif self.action_mode == "delta":
            magnet_values = self.get_magnets()
            self.set_magnets(magnet_values + action)

        # Run the simulation
        self.update_accelerator()

        # Build observation
        observation = {
            "beam": self.get_beam_parameters().astype("float32"),
            "magnets": self.get_magnets().astype("float32"),
            "target": self.target_beam
        }
        observation.update(self.get_accelerator_observation())

        # Compute reward
        current_beam = self.get_beam_parameters()

        # For readibility of the rewards below
        cb = current_beam
        ib = self.initial_screen_beam
        pb = self.previous_beam
        tb = self.target_beam

        on_screen_reward = -(not self.is_beam_on_screen())
        time_reward = -1
        if self.reward_mode == "differential":
            mu_x_reward = (abs(cb[0] - tb[0]) - abs(pb[0] - tb[0])) / abs(ib[0] - tb[0])
            sigma_x_reward = (abs(cb[1] - tb[1]) - abs(pb[1] - tb[1])) / abs(ib[1] - tb[1])
            mu_y_reward = (abs(cb[2] - tb[2]) - abs(pb[2] - tb[2])) / abs(ib[2] - tb[2])
            sigma_y_reward = (abs(cb[3] - tb[3]) - abs(pb[3] - tb[3])) / abs(ib[3] - tb[3])
        elif self.reward_mode == "feedback":
            mu_x_reward = - abs((cb[0] - tb[0]) / (ib[0] - tb[0]))
            sigma_x_reward = - (cb[1] - tb[1]) / (ib[1] - tb[1])
            mu_y_reward = - abs((cb[2] - tb[2]) / (ib[2] - tb[2]))
            sigma_y_reward = - (cb[3] - tb[3]) / (ib[3] - tb[3])

        reward = 1 * on_screen_reward + 1 * mu_x_reward + 1 * sigma_x_reward + 1 * mu_y_reward + 1 * sigma_y_reward
        reward = self.w_on_screen * on_screen_reward + self.w_mu_x * mu_x_reward + self.w_sigma_x * sigma_x_reward + self.w_mu_y * mu_y_reward + self.w_sigma_y * sigma_y_reward * self.w_time * time_reward
        reward = float(reward)

        # Figure out if reach good enough beam (done)
        done = bool(np.all(np.abs(cb) < self.target_beam_threshold))

        info = {
            "mu_x_reward": mu_x_reward,
            "mu_y_reward": mu_y_reward,
            "on_screen_reward": on_screen_reward,
            "sigma_x_reward": sigma_x_reward,
            "sigma_y_reward": sigma_y_reward,
            "time_reward": time_reward,
        }
        info.update(self.get_accelerator_info())
        
        self.previous_beam = current_beam

        return observation, reward, done, info
    
    def render(self, mode="human"):
        assert mode == "rgb_array" or mode == "human"

        binning = self.get_binning()
        pixel_size = self.get_pixel_size()
        resolution = self.get_screen_resolution()
        
        # Read screen image and make 8-bit RGB
        img = self.get_beam_image()
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:,:,np.newaxis], 3, axis=-1)

        # Redraw beam image as if it were binning = 4
        render_resolution = (resolution * binning / 4).astype("int")
        img = cv2.resize(img, render_resolution)

        # Draw beam ellipse
        beam = self.get_beam_parameters()
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(beam[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(beam[1] / pixel_size_b4[0])
        e_pos_y = int(-beam[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(beam[3] / pixel_size_b4[1])
        red = (0, 0, 255)
        img = cv2.ellipse(img, (e_pos_x,e_pos_y), (e_width_x,e_width_y), 0, 0, 360, red, 2)
        
        # Adjust aspect ratio
        new_width = int(img.shape[1] * pixel_size_b4[0] / pixel_size_b4[1])
        img = cv2.resize(img, (new_width,img.shape[0]))

        # Add magnet values
        magnets = self.get_magnets()
        padding = np.full((int(img.shape[0]*0.18),img.shape[1],3), fill_value=255, dtype=np.uint8)
        img = np.vstack([img, padding])
        black = (0, 0, 0)
        img = cv2.putText(img, f"Q1={magnets[0]:.2f}", (15,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"Q2={magnets[1]:.2f}", (215,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"CV={magnets[2]*1e3:.2f}", (415,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"Q3={magnets[3]:.2f}", (615,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"CH={magnets[4]*1e3:.2f}", (15,585), cv2.FONT_HERSHEY_SIMPLEX, 1, black)

        if mode == "human":
            cv2.imshow("ARES EA", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def is_beam_on_screen(self):
        """
        Return `True` when the beam is on the screen and `False` when it isn't.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError
    
    def setup_accelerator(self):
        """
        Prepare the accelerator for use with the environment. Should mostly be used for setting up
        simulations.

        Override with backend-specific imlementation. Optional.
        """
    
    def get_magnets(self):
        """
        Return the magnet values as a NumPy array in order as the magnets appear in the accelerator.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def set_magnets(self, magnets):
        """
        Set the magnets to the given values.

        The argument `magnets` will be passed as a NumPy array in the order the magnets appear in
        the accelerator.

        When applicable, this method should block until the magnet values are acutally set!

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def reset_accelerator(self):
        """
        Code that should set the accelerator up for a new episode. Run when the `reset` is called.

        Mostly meant for simulations to switch to a new incoming beam / misalignments or simular
        things.

        Override with backend-specific imlementation. Optional.
        """
    
    def update_accelerator(self):
        """
        Update accelerator metrics for later use. Use this to run the simulation or cache the beam
        image.

        Override with backend-specific imlementation. Optional.
        """
    
    def get_beam_parameters(self):
        """
        Get the beam parameters measured on the diagnostic screen as NumPy array grouped by
        dimension (e.g. mu_x, sigma_x, mu_y, sigma_y).

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError
    
    def get_incoming_parameters(self):
        """
        Get all physical beam parameters of the incoming beam as NumPy array in order energy, mu_x,
        mu_xp, mu_y, mu_yp, sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s, sigma_p.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_misalignments(self):
        """
        Get misalignments of the quadrupoles and the diagnostic screen as NumPy array in order
        AREAMQZM1.misalignment.x, AREAMQZM1.misalignment.y, AREAMQZM2.misalignment.x,
        AREAMQZM2.misalignment.y, AREAMQZM3.misalignment.x, AREAMQZM3.misalignment.y,
        AREABSCR1.misalignment.x, AREABSCR1.misalignment.y.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_beam_image(self):
        """
        Retreive the beam image as a 2-dimensional NumPy array.

        Note that if reading the beam image is expensive, it is best to cache the image in the
        `update_accelerator` method and the read the cached variable here.

        Ideally, the pixel values should look somewhat similar to the 12-bit values from the real
        screen camera.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_binning(self):
        """
        Return binning currently set on the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError
    
    def get_screen_resolution(self):
        """
        Return (binned) resolution of the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError
    
    def get_pixel_size(self):
        """
        Return the (binned) size of the area on the diagnostic screen covered by one pixel as NumPy
        array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_accelerator_observation_space(self):
        """
        Return a dictionary of aditional observation spaces for observations from the accelerator
        backend, e.g. incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}

    def get_accelerator_observation(self):
        """
        Return a dictionary of aditional observations from the accelerator backend, e.g. incoming
        beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}

    def get_accelerator_info(self):
        """
        Return a dictionary of aditional info from the accelerator backend, e.g. incoming beam and
        misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}


class ARESEACheetah(ARESEA):

    def __init__(
        self,
        incoming_mode="random",
        incoming_values=None,
        misalignment_mode="random",
        misalignment_values=None,
        action_mode="direct",
        magnet_init_mode="zero",
        magnet_init_values=None,
        reward_mode="differential",
        target_beam_mode="random",
        target_beam_values=None,
        target_beam_threshold=3.3198e-6,
        w_mu_x=1.0,
        w_mu_y=1.0,
        w_on_screen=1.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
        w_time=1.0,
    ):
        super().__init__(
            action_mode,
            magnet_init_mode,
            magnet_init_values,
            reward_mode,
            target_beam_mode,
            target_beam_values,
            target_beam_threshold,
            w_mu_x,
            w_mu_y,
            w_on_screen,
            w_sigma_x,
            w_sigma_y,
            w_time
        )

        self.incoming_mode = incoming_mode
        self.incoming_values = incoming_values
        self.misalignment_mode = misalignment_mode
        self.magnet_init_values = misalignment_values

        # Create particle simulation
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

    def is_beam_on_screen(self):
        screen = self.simulation.AREABSCR1
        beam_position = np.array([screen.read_beam.mu_x, screen.read_beam.mu_y])
        limits = np.array(screen.resolution) / 2 * np.array(screen.pixel_size)
        return np.all(np.abs(beam_position) < limits)
    
    def get_magnets(self):
        return np.array([
            self.simulation.AREAMQZM1.k1,
            self.simulation.AREAMQZM2.k1,
            self.simulation.AREAMCVM1.angle,
            self.simulation.AREAMQZM3.k1,
            self.simulation.AREAMCHM1.angle
        ])

    def set_magnets(self, magnets):
        print(f"Setting magnets {magnets}")
        self.simulation.AREAMQZM1.k1 = magnets[0]
        self.simulation.AREAMQZM2.k1 = magnets[1]
        self.simulation.AREAMCVM1.angle = magnets[2]
        self.simulation.AREAMQZM3.k1 = magnets[3]
        self.simulation.AREAMCHM1.angle = magnets[4]

    def reset_accelerator(self):
        # New domain randomisation
        if self.incoming_mode == "constant":
            incoming_parameters = self.incoming_values
        elif self.incoming_mode == "random":
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

        if self.misalignment_mode == "constant":
            misalignments = self.misalignment_values
        if self.misalignment_mode == "random":
            misalignments = self.observation_space["misalignments"].sample()
        self.simulation.AREAMQZM1.misalignment = misalignments[0:2]
        self.simulation.AREAMQZM2.misalignment = misalignments[2:4]
        self.simulation.AREAMQZM3.misalignment = misalignments[4:6]
        self.simulation.AREABSCR1.misalignment = misalignments[6:8]
    
    def update_accelerator(self):
        self.simulation(self.incoming)
    
    def get_beam_parameters(self):
        return np.array([
            self.simulation.AREABSCR1.read_beam.mu_x,
            self.simulation.AREABSCR1.read_beam.sigma_x,
            self.simulation.AREABSCR1.read_beam.mu_y,
            self.simulation.AREABSCR1.read_beam.sigma_y
        ])
    
    def get_incoming_parameters(self):
        # Parameters of incoming are typed out to guarantee their order, as the
        # order would not be guaranteed creating np.array from dict.
        return np.array([
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
        ])

    def get_misalignments(self):
        return np.array([
            self.simulation.AREAMQZM1.misalignment[0],
            self.simulation.AREAMQZM1.misalignment[1],
            self.simulation.AREAMQZM2.misalignment[0],
            self.simulation.AREAMQZM2.misalignment[1],
            self.simulation.AREAMQZM3.misalignment[0],
            self.simulation.AREAMQZM3.misalignment[1],
            self.simulation.AREABSCR1.misalignment[0],
            self.simulation.AREABSCR1.misalignment[1]
        ], dtype=np.float32)

    def get_beam_image(self):
        # Beam image to look like real image by dividing by goodlooking number and scaling to 12 bits)
        return self.simulation.AREABSCR1.reading / 1e9 * 2**12

    def get_binning(self):
        return np.array(self.simulation.AREABSCR1.binning)
    
    def get_screen_resolution(self):
        return np.array(self.simulation.AREABSCR1.resolution) / self.get_binning()
    
    def get_pixel_size(self):
        return np.array(self.simulation.AREABSCR1.pixel_size) * self.get_binning()

    def get_accelerator_observation_space(self):
        return {
            "incoming": spaces.Box(
                low=np.array([80e6, -1e-3, -1e-4, -1e-3, -1e-4, 1e-5, 1e-6, 1e-5, 1e-6, 1e-6, 1e-4], dtype=np.float32),
                high=np.array([160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3], dtype=np.float32)
            ),
            "misalignments": spaces.Box(low=-400e-6, high=400e-6, shape=(8,)),
        }

    def get_accelerator_observation(self):
        return {
            "incoming": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
        }


class ARESEADOOCS(ARESEA):
    
    def __init__(
        self,
        dummy=True,
        action_mode="direct",
        magnet_init_mode="zero",
        magnet_init_values=None,
        reward_mode="differential",
        target_beam_mode="random",
        target_beam_values=None,
        target_beam_threshold=3.3198e-6,
        w_mu_x=1.0,
        w_mu_y=1.0,
        w_on_screen=1.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
        w_time=1.0,
    ):
        # Import pydoocs only when this class is loaded and choose dummypydoocs if requested
        global pydoocs
        if dummy:
            import dummypydoocs as pydoocs
        else:
            import pydoocs

        super().__init__(
            action_mode,
            magnet_init_mode,
            magnet_init_values,
            reward_mode,
            target_beam_mode,
            target_beam_values,
            target_beam_threshold,
            w_mu_x,
            w_mu_y,
            w_on_screen,
            w_sigma_x,
            w_sigma_y,
            w_time
        )

    def is_beam_on_screen(self):
        return True # TODO find better logic

    def get_magnets(self):
        return np.array([
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.RBV")["data"],
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.RBV")["data"],
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.RBV")["data"] / 1000,
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.RBV")["data"],
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.RBV")["data"] / 1000
        ])
    
    def set_magnets(self, magnets):
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.SP", magnets[0])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.SP", magnets[1])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.SP", magnets[2] * 1000)
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.SP", magnets[3])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.SP", magnets[4] * 1000)

        # Wait until magnets have reached their setpoints
        
        time.sleep(3.0) # Wait for magnets to realise they received a command

        magnets = ["AREAMQZM1", "AREAMQZM2", "AREAMCVM1", "AREAMQZM3", "AREAMCHM1"]

        are_busy = [True] * 5
        are_ps_on = [True] * 5
        while any(are_busy) or not all(are_ps_on):
            are_busy = [pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/BUSY")["data"] for magnet in magnets]
            are_ps_on = [pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/PS_ON")["data"] for magnet in magnets]

    def update_accelerator(self):
        self.beam_image = self.capture_clean_beam_image()

    def get_beam_parameters(self):
        img = self.get_beam_image()
        pixel_size = self.get_pixel_size()

        parameters = {}
        for axis, direction in zip([0,1], ["x","y"]):
            projection = img.sum(axis=axis)
            minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
            filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

            half_values, = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2
            else:
                fwhm_pixel = 42     # TODO figure out what to do with these
                center_pixel = 42

            parameters[f"mu_{direction}"] = (center_pixel - len(filtered) / 2) * pixel_size[axis]
            parameters[f"sigma_{direction}"] = fwhm_pixel / 2.355 * pixel_size[axis]
            
        parameters["mu_y"] = -parameters["mu_y"]

        return np.array([
            parameters["mu_x"],
            parameters["sigma_x"],
            parameters["mu_y"],
            parameters["sigma_y"]
        ])

    def get_beam_image(self):
        return self.beam_image

    def get_binning(self):
        return np.array((
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL")["data"],
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL")["data"]
        ))

    def get_screen_resolution(self):
        return np.array([
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/WIDTH")["data"],
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/HEIGHT")["data"]
        ])
    
    def get_pixel_size(self):
        return np.array([
            abs(pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/X.POLY_SCALE")["data"][2]) / 1000,
            abs(pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/Y.POLY_SCALE")["data"][2]) / 1000
        ]) * self.get_binning()

    def capture_clean_beam_image(self, average=5):
        """
        Capture a clean image of the beam from the screen using `average` images with beam on and
        `average` images of the background and then removing the background.
        
        Saves the image to a property of the object.
        """
         # Laser off
        self.set_cathode_laser(False)
        background_images = self.capture_interval(n=average, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        self.set_cathode_laser(True)
        beam_images = self.capture_interval(n=average, dt=0.1)
        median_beam = np.median(beam_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16-1)
        flipped = np.flipud(removed)
        
        return flipped
    
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

    def set_cathode_laser(self, setto):
        """Sets the bool switch of the cathode laser event to `setto` and waits a second."""
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)


def read_from_yaml(path):
    with open(f"{path}.yaml", "r") as f:
        data = yaml.parse(f.read())
    return data


def save_to_yaml(data, path):
    with open(f"{path}.yaml", "w") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    main()
