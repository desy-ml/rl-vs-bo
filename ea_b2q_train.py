from functools import partial

import cheetah
import gym
import numpy as np
import wandb
import yaml
from gym import spaces
from gym.wrappers import (
    FilterObservation,
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from ARESlatticeStage3v1_9 import cell as ares_lattice
from utils import save_config


def main():
    config = {
        "action_mode": "delta",
        "gamma": 0.99,
        "filter_observation": None,
        "frame_stack": None,
        "incoming_mode": "random",
        "incoming_values": None,
        "magnet_init_mode": "constant",
        "magnet_init_values": np.zeros(4),
        "misalignment_mode": "random",
        "misalignment_values": None,
        "n_envs": 40,
        "normalize_observation": True,
        "normalize_reward": True,
        "rescale_action": (-1, 1),
        "reward_mode": "differential",
        "sb3_device": "auto",
        "target_threshold": 3.3198e-6,
        "threshold_hold": 5,
        "time_limit": 25,
        "vec_env": "subproc",
        "w_done": 10.0,
        "w_movement": 1.0,
        "w_threshold": 0.0,
        "w_time": 0.0,
    }

    train(config)


def train(config):
    # Setup wandb
    wandb.init(
        project="ares-ea-beam-2-quad",
        entity="msk-ipc",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
    )
    config["wandb_run_name"] = wandb.run.name

    # Setup environments
    if config["vec_env"] == "dummy":
        env = DummyVecEnv([partial(make_env, config) for _ in range(config["n_envs"])])
    elif config["vec_env"] == "subproc":
        env = SubprocVecEnv(
            [partial(make_env, config) for _ in range(config["n_envs"])]
        )
    else:
        raise ValueError(f"Invalid value \"{config['vec_env']}\" for dummy")
    eval_env = DummyVecEnv([partial(make_env, config, record_video=False)])

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
        n_steps=100,
        batch_size=100,
        verbose=2,
    )

    model.learn(
        total_timesteps=int(2e6),
        eval_env=eval_env,
        eval_freq=500,
        callback=WandbCallback(),
    )

    model.save(f"models/{wandb.run.name}/model")
    if config["normalize_observation"] or config["normalize_reward"]:
        env.save(f"models/{wandb.run.name}/vec_normalize.pkl")
    save_config(config, f"models/{wandb.run.name}/config")


def make_env(config, record_video=False):
    env = ARESEACheetah(
        incoming_mode=config["incoming_mode"],
        incoming_values=config["incoming_values"],
        misalignment_mode=config["misalignment_mode"],
        misalignment_values=config["misalignment_values"],
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=config["magnet_init_values"],
        reward_mode=config["reward_mode"],
        target_threshold=config["target_threshold"],
        threshold_hold=config["threshold_hold"],
        w_done=config["w_done"],
        w_movement=config["w_movement"],
        w_threshold=config["w_threshold"],
        w_time=config["w_time"],
    )
    if config["filter_observation"] is not None:
        env = FilterObservation(env, config["filter_observation"])
    if config["time_limit"] is not None:
        env = TimeLimit(env, config["time_limit"])
    env = FlattenObservation(env)
    if config["frame_stack"] is not None:
        env = FrameStack(env, config["frame_stack"])
    if config["rescale_action"] is not None:
        env = RescaleAction(
            env, config["rescale_action"][0], config["rescale_action"][1]
        )
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
        How actions work. Choose `"direct"`, `"direct_unidirectional_quads"` or
        `"delta"`.
    magnet_init_mode : str
        Magnet initialisation on `reset`. Set to `None`, `"random"` or `"constant"`. The
        `"constant"` setting requires `magnet_init_values` to be set.
    magnet_init_values : np.ndarray
        Values to set magnets to on `reset`. May only be set when `magnet_init_mode` is
        set to `"constant"`.
    reward_mode : str
        How to compute the reward. Choose from `"feedback"` or `"differential"`.
    target_beam_mode : str
        Setting of target beam on `reset`. Choose from `"constant"` or `"random"`. The
        `"constant"` setting requires `target_beam_values` to be set.
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 2}

    def __init__(
        self,
        action_mode="direct",
        include_beam_image_in_info=True,
        magnet_init_mode=None,
        magnet_init_values=None,
        reward_mode="differential",
        target_threshold=3.3198e-6,
        threshold_hold=1,
        w_done=1.0,
        w_movement=1.0,
        w_threshold=1.0,
        w_time=1.0,
    ):
        self.action_mode = action_mode
        self.include_beam_image_in_info = include_beam_image_in_info
        self.magnet_init_mode = magnet_init_mode
        self.magnet_init_values = magnet_init_values
        self.reward_mode = reward_mode
        self.target_threshold = target_threshold
        self.threshold_hold = threshold_hold
        self.w_done = w_done
        self.w_movement = w_movement
        self.w_threshold = w_threshold
        self.w_time = w_time

        # Create action space
        if self.action_mode == "direct":
            self.action_space = spaces.Box(low=-6.1782e-3, high=6.1782e-3, shape=(4,))
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=-6.1782e-3 * 0.1, high=6.1782e-3 * 0.1, shape=(4,)
            )
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

        # Create observation space
        obs_space_dict = {
            "beam_movements": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6,),
            ),
            "magnets": spaces.Box(low=-6.1782e-3, high=6.1782e-3, shape=(4,)),
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
        elif self.magnet_init_mode is None:
            pass  # This really is intended to do nothing
        else:
            raise ValueError(
                f'Invalid value "{self.magnet_init_mode}" for magnet_init_mode'
            )

        # Update anything in the accelerator (mainly for running simulations)
        self.update_accelerator()

        self.initial_beam_movements = self.measure_beam_movements()
        self.previous_beam_movements = self.initial_beam_movements
        self.is_in_threshold_history = []
        self.steps_taken = 0

        observation = {
            "beam_movements": self.initial_beam_movements.astype("float32"),
            "magnets": self.get_magnets().astype("float32"),
        }
        observation.update(self.get_accelerator_observation())

        return observation

    def step(self, action):
        # Perform action
        if self.action_mode == "direct":
            self.set_magnets(action)
        elif self.action_mode == "direct_unidirectional_quads":
            self.set_magnets(action)
        elif self.action_mode == "delta":
            magnet_values = self.get_magnets()
            self.set_magnets(magnet_values + action)
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

        # Run the simulation
        self.update_accelerator()

        current_beam_movements = self.measure_beam_movements()
        self.steps_taken += 1

        # Build observation
        observation = {
            "beam_movements": current_beam_movements.astype("float32"),
            "magnets": self.get_magnets().astype("float32"),
        }
        observation.update(self.get_accelerator_observation())

        # For readibility in computations below
        cb = max(np.sum(np.abs(current_beam_movements)), 1e-10)
        ib = max(np.sum(np.abs(self.initial_beam_movements)), 1e-10)
        pb = max(np.sum(np.abs(self.previous_beam_movements)), 1e-10)

        # Compute if done (beam within threshold for a certain time)
        threshold = self.target_threshold
        is_in_threshold = (np.abs(cb) < threshold).all()
        self.is_in_threshold_history.append(is_in_threshold)
        is_stable_in_threshold = bool(
            np.array(self.is_in_threshold_history[-self.threshold_hold :]).all()
        )
        done = is_stable_in_threshold and len(self.is_in_threshold_history) > 5

        # Compute reward
        time_reward = -1
        done_reward = done * (25 - self.steps_taken) / 25
        if self.reward_mode == "differential":
            movement_reward = (pb - cb) / ib
        elif self.reward_mode == "feedback":
            movement_reward = -cb / ib
        else:
            raise ValueError(f'Invalid value "{self.reward_mode}" for reward_mode')

        reward = 0
        reward += self.w_movement * movement_reward
        reward += self.w_time * time_reward
        reward += self.w_threshold * is_in_threshold
        reward += self.w_done * done_reward
        reward = float(reward)

        # Put together info
        info = {
            "binning": self.get_binning(),
            "movement_reward": movement_reward,
            "pixel_size": self.get_pixel_size(),
            "screen_resolution": self.get_screen_resolution(),
            "time_reward": time_reward,
        }
        info.update(self.get_accelerator_info())

        self.previous_beam_movements = current_beam_movements

        return observation, reward, done, info

    def render(self, mode="human"):
        raise NotImplementedError

    def measure_beam_movements(self):
        """
        Measure how much the beam moves when each of the quadrupoles is turned up (e.g.
        q1_dx, q1_dy, q2_dx, q2_dy, q3_dx, q3_dy).
        """
        return np.concatenate(
            [
                self.measure_beam_movement(quad)
                for quad in ["AREAMQZM1", "AREAMQZM2", "AREAMQZM3"]
            ]
        )

    def measure_beam_movement(self, quadrupole):
        """
        Measure how much the beam moves when a `quadrupole` is turned up (e.g. q1_dx,
        q1_dy, q2_dx, q2_dy, q3_dx, q3_dy).
        """
        self.set_quadrupole(quadrupole, 0.0)
        self.update_accelerator()
        beam_off = self.get_beam_parameters()

        self.set_quadrupole(quadrupole, 3.0)
        self.update_accelerator()
        beam_on = self.get_beam_parameters()

        self.set_quadrupole(quadrupole, 0.0)

        beam_movement = np.array([beam_on[0] - beam_off[0], beam_on[2] - beam_off[2]])

        return beam_movement

    def setup_accelerator(self):
        """
        Prepare the accelerator for use with the environment. Should mostly be used for
        setting up simulations.

        Override with backend-specific imlementation. Optional.
        """

    def get_magnets(self):
        """
        Return the magnet values as a NumPy array in order as the magnets appear in the
        accelerator.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def set_magnets(self, magnets):
        """
        Set the magnets to the given values.

        The argument `magnets` will be passed as a NumPy array in the order the magnets
        appear in the accelerator.

        When applicable, this method should block until the magnet values are acutally
        set!

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def set_quadrupole(self, name, strength):
        """
        Write `strength` to the quadrupole `name`.

        Override with backend-specific imlementation.
        """
        raise NotImplementedError

    def reset_accelerator(self):
        """
        Code that should set the accelerator up for a new episode. Run when the `reset`
        is called.

        Mostly meant for simulations to switch to a new incoming beam / misalignments or
        simular things.

        Override with backend-specific imlementation. Optional.
        """

    def update_accelerator(self):
        """
        Update accelerator metrics for later use. Use this to run the simulation or
        cache the beam image.

        Override with backend-specific imlementation. Optional.
        """

    def get_beam_parameters(self):
        """
        Get the beam parameters measured on the diagnostic screen as NumPy array grouped
        by dimension (e.g. mu_x, sigma_x, mu_y, sigma_y).

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def get_incoming_parameters(self):
        """
        Get all physical beam parameters of the incoming beam as NumPy array in order
        energy, mu_x, mu_xp, mu_y, mu_yp, sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s,
        sigma_p.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_misalignments(self):
        """
        Get misalignments of the quadrupoles and the diagnostic screen as NumPy array in
        order AREAMQZM1.misalignment.x, AREAMQZM1.misalignment.y,
        AREAMQZM2.misalignment.x, AREAMQZM2.misalignment.y, AREAMQZM3.misalignment.x,
        AREAMQZM3.misalignment.y, AREABSCR1.misalignment.x, AREABSCR1.misalignment.y.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_beam_image(self):
        """
        Retreive the beam image as a 2-dimensional NumPy array.

        Note that if reading the beam image is expensive, it is best to cache the image
        in the `update_accelerator` method and the read the cached variable here.

        Ideally, the pixel values should look somewhat similar to the 12-bit values from
        the real screen camera.

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
        Return the (binned) size of the area on the diagnostic screen covered by one
        pixel as NumPy array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_accelerator_observation_space(self):
        """
        Return a dictionary of aditional observation spaces for observations from the
        accelerator backend, e.g. incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}

    def get_accelerator_observation(self):
        """
        Return a dictionary of aditional observations from the accelerator backend, e.g.
        incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}

    def get_accelerator_info(self):
        """
        Return a dictionary of aditional info from the accelerator backend, e.g.
        incoming beam and misalignments in simulation.

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
        magnet_init_mode=None,
        magnet_init_values=None,
        reward_mode="differential",
        target_threshold=3.3198e-6,
        threshold_hold=1,
        w_done=1.0,
        w_movement=1.0,
        w_threshold=1.0,
        w_time=1.0,
    ):
        super().__init__(
            action_mode=action_mode,
            magnet_init_mode=magnet_init_mode,
            magnet_init_values=magnet_init_values,
            reward_mode=reward_mode,
            target_threshold=target_threshold,
            threshold_hold=threshold_hold,
            w_done=w_done,
            w_movement=w_movement,
            w_threshold=w_threshold,
            w_time=w_time,
        )

        self.incoming_mode = incoming_mode
        self.incoming_values = incoming_values
        self.misalignment_mode = misalignment_mode
        self.misalignment_values = misalignment_values

        # Create particle simulation
        self.simulation = cheetah.Segment.from_ocelot(
            ares_lattice, warnings=False, device="cpu"
        ).subcell("ARLIBSCR2", "AREABSCR1")
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
        return np.array(
            [
                self.simulation.ARLIMCHM1.angle,
                self.simulation.ARLIMCVM1.angle,
                self.simulation.ARLIMCHM2.angle,
                self.simulation.ARLIMCVM2.angle,
            ]
        )

    def set_magnets(self, magnets):
        self.simulation.ARLIMCHM1.angle = magnets[0]
        self.simulation.ARLIMCVM1.angle = magnets[1]
        self.simulation.ARLIMCHM2.angle = magnets[2]
        self.simulation.ARLIMCVM2.angle = magnets[3]

    def set_quadrupole(self, name, strength):
        getattr(self.simulation, name).k1 = strength

    def reset_accelerator(self):
        # New domain randomisation
        if self.incoming_mode == "constant":
            incoming_parameters = self.incoming_values
        elif self.incoming_mode == "random":
            incoming_parameters = self.observation_space["incoming"].sample()
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
            misalignments = self.observation_space["misalignments"].sample()
        else:
            raise ValueError(
                f'Invalid value "{self.misalignment_mode}" for misalignment_mode'
            )
        self.simulation.AREAMQZM1.misalignment = misalignments[0:2]
        self.simulation.AREAMQZM2.misalignment = misalignments[2:4]
        self.simulation.AREAMQZM3.misalignment = misalignments[4:6]
        self.simulation.AREABSCR1.misalignment = misalignments[6:8]

    def update_accelerator(self):
        self.simulation(self.incoming)

    def get_beam_parameters(self):
        return np.array(
            [
                self.simulation.AREABSCR1.read_beam.mu_x,
                self.simulation.AREABSCR1.read_beam.sigma_x,
                self.simulation.AREABSCR1.read_beam.mu_y,
                self.simulation.AREABSCR1.read_beam.sigma_y,
            ]
        )

    def get_incoming_parameters(self):
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

    def get_misalignments(self):
        return np.array(
            [
                self.simulation.AREAMQZM1.misalignment[0],
                self.simulation.AREAMQZM1.misalignment[1],
                self.simulation.AREAMQZM2.misalignment[0],
                self.simulation.AREAMQZM2.misalignment[1],
                self.simulation.AREAMQZM3.misalignment[0],
                self.simulation.AREAMQZM3.misalignment[1],
                self.simulation.AREABSCR1.misalignment[0],
                self.simulation.AREABSCR1.misalignment[1],
            ],
            dtype=np.float32,
        )

    def get_beam_image(self):
        # Beam image to look like real image by dividing by goodlooking number and
        # scaling to 12 bits)
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
            "misalignments": spaces.Box(low=-2e-3, high=2e-3, shape=(8,)),
        }

    def get_accelerator_observation(self):
        return {
            "incoming": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
        }


if __name__ == "__main__":
    main()
