import argparse
from functools import partial

import cheetah
import cv2
import gym
from gym import spaces
from gym.wrappers import FilterObservation, FlattenObservation, FrameStack, RecordVideo, RescaleAction, TimeLimit
import numpy as np
import yaml
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from ARESlatticeStage3v1_9 import cell as ares_lattice
from utils import CheckpointCallback, FilterAction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_type", type=str, default="direct", choices=["direct","delta"])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--filter_action", nargs="+", type=int, default=[0,1,2,3,4])
    parser.add_argument("--filter_observation", nargs="+", type=str, default=["beam","incoming","magnets","misalignments"])
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--incoming", type=str, default="constant", choices=["constant","random"])
    parser.add_argument("--magnet_init", type=str, default="zero", choices=["zero","random"])
    parser.add_argument("--misalignments", type=str, default="constant", choices=["constant","random"])
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--normalize_observation", action="store_true", default=False)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--reward_method", type=str, default="differential", choices=["differential","feedback"])
    parser.add_argument("--sb3_device", type=str, default="auto")
    parser.add_argument("--target_beam_mode", type=str, default="zero", choices=["zero","random"])
    parser.add_argument("--target_beam_tolerance", type=float, default=3.3198e-6)
    parser.add_argument("--time_limit", type=int, default=50)
    parser.add_argument("--total_timesteps", type=int, default=800000)
    parser.add_argument("--quad_action", type=str, default="symmetric", choices=["symmetric","oneway"])
    parser.add_argument("--vec_env", type=str, default="dummy", choices=["dummy","subproc"])
    parser.add_argument("--w_mu_x", type=float, default=1.0)
    parser.add_argument("--w_mu_y", type=float, default=1.0)
    parser.add_argument("--w_on_screen", type=float, default=1.0)
    parser.add_argument("--w_sigma_x", type=float, default=1.0)
    parser.add_argument("--w_sigma_y", type=float, default=1.0)
    parser.add_argument("--w_time", type=float, default=1.0)

    args = parser.parse_args()
    return vars(args)


def main():
    # Parse config and generate config elements that result from given arguments
    config = parse_args()
    if config["incoming"] == "constant":
        config["incoming_parameters"] = ARESEA().observation_space["incoming"].sample()
    if config["misalignments"] == "constant":
        config["misalignment_values"] = ARESEA().observation_space["misalignments"].sample()
    
    # Setup wandb
    wandb.init(
        project="ares-ea-v2",
        entity="msk-ipc",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config
    )
    config["wandb_run_name"] = wandb.run.name

    # Setup environment
    if config["vec_env"] == "dummy":
        env = DummyVecEnv([partial(make_env, config) for _ in range(config["n_envs"])])
        eval_env = DummyVecEnv([partial(make_eval_env, config)])
    elif config["vec_env"] == "subproc":
        env = SubprocVecEnv([partial(make_env, config) for _ in range(config["n_envs"])])
        eval_env = SubprocVecEnv([partial(make_eval_env, config)])
    else:
        raise ValueError(f"Invalid value \"{config['vec_env']}\" for dummy")

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
        tensorboard_log=f"log/{wandb.run.name}",
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


def make_env(config):
    env = ARESEA(**config)
    env = FilterObservation(env, config["filter_observation"])
    env = FilterAction(env, config["filter_action"], replace=0)
    env = TimeLimit(env, max_episode_steps=config["time_limit"])
    env = FlattenObservation(env)
    env = FrameStack(env, config["frame_stack"])
    env = RescaleAction(env, -3, 3)
    env = Monitor(env)
    return env


def make_eval_env(config):
    env = ARESEA(**config)
    env = FilterObservation(env, config["filter_observation"])
    env = FilterAction(env, config["filter_action"], replace=0)
    env = TimeLimit(env, max_episode_steps=config["time_limit"])
    env = RecordVideo(env, video_folder=f"recordings/{config['wandb_run_name']}")
    env = FlattenObservation(env)
    env = FrameStack(env, config["frame_stack"])
    env = RescaleAction(env, -3, 3)
    env = Monitor(env)
    return env


class ARESEA(gym.Env):

    metadata = {
        "render.modes": ["rgb_array"],
        "video.frames_per_second": 2
    }

    def __init__(
        self,
        action_type="direct",
        is_fully_observable=False,
        incoming="random",
        incoming_parameters=None,
        magnet_init="zero",
        misalignments="random",
        misalignment_values=None,
        reward_method="differential",
        quad_action="symmetric",
        target_beam_mode="zero",
        target_beam_tolerance=3.3198e-6,
        w_mu_x=1.0,
        w_mu_y=1.0,
        w_on_screen=1.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
        w_time=1.0
    ):
        self.action_type = action_type
        self.is_fully_observable = is_fully_observable
        self.incoming = incoming
        self.incoming_parameters = incoming_parameters
        self.magnet_init = magnet_init
        self.misalignments = misalignments
        self.misalignment_values = misalignment_values
        self.quad_action = quad_action
        self.reward_method = reward_method
        self.target_beam_mode = target_beam_mode
        self.target_beam_tolerance = target_beam_tolerance
        self.w_mu_x = w_mu_x
        self.w_mu_y = w_mu_y
        self.w_on_screen = w_on_screen
        self.w_sigma_x = w_sigma_x
        self.w_sigma_y = w_sigma_y
        self.w_time = w_time

        # Setup observation and action spaces
        if quad_action == "symmetric":
            magnet_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
            )
        elif quad_action == "oneway":
            magnet_space = spaces.Box(
                low=np.array([0, -72, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 0, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
            )
        else:
            raise ValueError(f"Invalid quad_action \"{self.quad_action}\"")

        # Create action space
        if self.action_type == "direct":
            self.action_space = magnet_space
        elif self.action_type == "delta":
            self.action_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32) * 0.1,
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32) * 0.1
            )
        else:
            raise ValueError(f"Invalid action_type \"{self.action_type}\"")

        # Create observation space
        obs_space_dict = {
            "beam": spaces.Box(
                low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
            ),
            "magnets": magnet_space,
            "target": spaces.Box(
                low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
            )
        }
        if self.is_fully_observable:
            obs_space_dict["incoming"] = spaces.Box(
                low=np.array([80e6, -1e-3, -1e-4, -1e-3, -1e-4, 1e-5, 1e-6, 1e-5, 1e-6, 1e-6, 1e-4], dtype=np.float32),
                high=np.array([160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3], dtype=np.float32)
            )
            obs_space_dict["misalignments"] = spaces.Box(low=-400e-6, high=400e-6, shape=(8,))
        self.observation_space = spaces.Dict(obs_space_dict)

        # Setup the accelerator (either simulation or the actual machine)
        self.setup_accelerator()
    
    def reset(self):
        if self.magnet_init == "zero":
            self.set_magnets(*np.zeros(5))
        elif self.magnet_init == "random":
            magnets = self.observation_space["magnets"].sample()
            self.set_magnets(*magnets)
        else:
            raise ValueError(f"Invalid value for magnet_init \"{self.magnet_init}\"")

        self.reset_accelerator()

        if self.target_beam_mode == "random":
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
        if self.is_fully_observable:
            observation["incoming"] = self.get_incoming_parameters()
            observation["misalignments"] = self.get_misalignments()

        return observation

    def step(self, action):
        # Perform action
        if self.action_type == "direct":
            self.set_magnets(*action)
        elif self.action_type == "delta":
            magnet_values = self.get_magnets()
            self.set_magnets((magnet_values + action))
        else:
            raise ValueError(f"Invalid action_type \"{self.action_type}\"")

        # Run the simulation
        self.update_accelerator()

        # Build observation
        observation = {
            "beam": self.get_beam_parameters().astype("float32"),
            "magnets": self.get_magnets().astype("float32"),
            "target": self.target_beam
        }
        if self.is_fully_observable:
            observation["incoming"] = self.get_incoming_parameters()
            observation["misalignments"] = self.get_misalignments()

        # Compute reward
        current_beam = self.get_beam_parameters()

        # For readibility of the rewards below
        cb = current_beam
        ib = self.initial_screen_beam
        pb = self.previous_beam
        tb = self.target_beam

        on_screen_reward = -(not self.is_beam_on_screen())
        time_reward = -1
        if self.reward_method == "differential":
            mu_x_reward = (abs(cb[0] - tb[0]) - abs(pb[0] - tb[0])) / abs(ib[0] - tb[0])
            sigma_x_reward = (abs(cb[1] - tb[1]) - abs(pb[1] - tb[1])) / abs(ib[1] - tb[1])
            mu_y_reward = (abs(cb[2] - tb[2]) - abs(pb[2] - tb[2])) / abs(ib[2] - tb[2])
            sigma_y_reward = (abs(cb[3] - tb[3]) - abs(pb[3] - tb[3])) / abs(ib[3] - tb[3])
        elif self.reward_method == "feedback":
            mu_x_reward = - abs((cb[0] - tb[0]) / (ib[0] - tb[0]))
            sigma_x_reward = - (cb[1] - tb[1]) / (ib[1] - tb[1])
            mu_y_reward = - abs((cb[2] - tb[2]) / (ib[2] - tb[2]))
            sigma_y_reward = - (cb[3] - tb[3]) / (ib[3] - tb[3])
        else:
            raise ValueError(f"Invalid reward method \"{self.reward_method}\"")

        reward = 1 * on_screen_reward + 1 * mu_x_reward + 1 * sigma_x_reward + 1 * mu_y_reward + 1 * sigma_y_reward
        reward = self.w_on_screen * on_screen_reward + self.w_mu_x * mu_x_reward + self.w_sigma_x * sigma_x_reward + self.w_mu_y * mu_y_reward + self.w_sigma_y * sigma_y_reward * self.w_time * time_reward
        reward = float(reward)

        # Figure out if reach good enough beam (done)
        done = bool(np.all(np.abs(cb) < self.target_beam_tolerance))

        info = {
            "incoming": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
            "mu_x_reward": mu_x_reward,
            "mu_y_reward": mu_y_reward,
            "on_screen_reward": on_screen_reward,
            "sigma_x_reward": sigma_x_reward,
            "sigma_y_reward": sigma_y_reward,
            "time_reward": time_reward,
        }
        
        self.previous_beam = current_beam

        return observation, reward, done, info
    
    def render(self, mode="human"):
        assert mode == "rgb_array" or mode == "human"

        # Read screen image and make 8-bit RGB
        img = self.get_beam_image()
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:,:,np.newaxis], 3, axis=-1)

        # Draw beam ellipse
        beam = self.get_beam_parameters()
        binning = self.get_binning()
        pixel_size = self.get_pixel_size() * binning
        resolution = self.get_screen_resolution() / binning
        e_pos_x = int(beam[0] / pixel_size[0] + resolution[0] / 2)
        e_width_x = int(beam[1] / pixel_size[0])
        e_pos_y = int(-beam[2] / pixel_size[1] + resolution[1] / 2)
        e_width_y = int(beam[3] / pixel_size[1])
        red = (0, 0, 255)
        img = cv2.ellipse(img, (e_pos_x,e_pos_y), (e_width_x,e_width_y), 0, 0, 360, red, 2)
        
        # Adjust aspect ration
        new_width = int(img.shape[1] * pixel_size[0] / pixel_size[1])
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
        screen = self.simulation.AREABSCR1
        beam_position = np.array([screen.read_beam.mu_x, screen.read_beam.mu_y])
        limits = np.array(screen.resolution) / 2 * np.array(screen.pixel_size)
        return np.all(np.abs(beam_position) < limits)
    
    def setup_accelerator(self):
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

        # If constant, set misalignments and incoming beam to passed values
        if self.incoming == "constant":
            self.incoming = cheetah.ParameterBeam.from_parameters(
                energy=self.incoming_parameters[0],
                mu_x=self.incoming_parameters[1],
                mu_xp=self.incoming_parameters[2],
                mu_y=self.incoming_parameters[3],
                mu_yp=self.incoming_parameters[4],
                sigma_x=self.incoming_parameters[5],
                sigma_xp=self.incoming_parameters[6],
                sigma_y=self.incoming_parameters[7],
                sigma_yp=self.incoming_parameters[8],
                sigma_s=self.incoming_parameters[9],
                sigma_p=self.incoming_parameters[10],
            )
        if self.misalignments == "constant":
            self.simulation.AREAMQZM1.misalignment = self.misalignment_values[0:2]
            self.simulation.AREAMQZM2.misalignment = self.misalignment_values[2:4]
            self.simulation.AREAMQZM3.misalignment = self.misalignment_values[4:6]
            self.simulation.AREABSCR1.misalignment = self.misalignment_values[6:8]
        if self.target_beam_mode == "zero":
            self.target_beam = np.zeros(4, dtype=np.float32)
    
    def get_magnets(self):
        return np.array([
            self.simulation.AREAMQZM1.k1,
            self.simulation.AREAMQZM2.k1,
            self.simulation.AREAMCVM1.angle,
            self.simulation.AREAMQZM3.k1,
            self.simulation.AREAMCHM1.angle
        ])

    def set_magnets(self, areamqzm1=0.0, areamqzm2=0.0, areamcvm1=0.0, areamqzm3=0.0, areamchm1=0.0):
        self.simulation.AREAMQZM1.k1 = areamqzm1
        self.simulation.AREAMQZM2.k1 = areamqzm2
        self.simulation.AREAMCVM1.angle = areamcvm1
        self.simulation.AREAMQZM3.k1 = areamqzm3
        self.simulation.AREAMCHM1.angle = areamchm1

    def reset_accelerator(self):
        # New domain randomisation
        if self.incoming == "random":
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
        if self.misalignments == "random":
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
        return np.array(self.simulation.AREABSCR1.resolution)
    
    def get_pixel_size(self):
        return np.array(self.simulation.AREABSCR1.pixel_size)


def save_to_yaml(data, path):
    with open(f"{path}.yaml", "w") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    main()
