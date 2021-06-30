import gym
from gym.spaces import Box
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
from numpy.random import uniform
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from accelerator_environments.envs.optimization import Optimizable


class ARESEA(gym.Env, Optimizable):
    """
    ARESEA tasks the agent with centering and focusing an electron beam on a diagnostic screen
    downstream from the experimental in ARES (Accelerator Research Experiment at SINBAD).
    
    To achieve beam focusing and centering, the agent may control a quadrupole triplet and two
    correctors, a vertical and a horizontal one. Rewards are given according to the improvement of
    the beam score computed as the sum of beam parameters :math:`(\mu_x, \mu_y, \sigma_x, \sigma_y)`.
    """

    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 5}
    
    binning = 4
    screen_resolution = (2448, 2040)
    pixel_size = (3.5488e-6, 2.5003e-6)


    observation_space = Box(low=np.array([-4e-3, -4e-3,    0,    0,   0, -10, -10, -10, -1e-3, -1e-3], dtype=np.float32),
                            high=np.array([4e-3,  4e-3, 4e-4, 4e-4, 1e5,  10,  10,  10,  1e-3,  1e-3], dtype=np.float32))
    action_space = Box(low=observation_space.low[-5:] * 0.1,
                       high=observation_space.high[-5:] * 0.1)

    optimization_space = Box(low=observation_space.low[-5:],
                             high=observation_space.high[-5:])
    
    target = np.array([0, 0, 0, 0])
    goal = np.array([1e-4, 1e-4, 5e-4, 5e-4])

    def __init__(self, objective_method="abs", reward_method="differential", target_translation=False):
        super().__init__()
        self.objective_method = objective_method
        self.reward_method = reward_method
        self.target_translation = target_translation

        self.magnets_changed = True
            
    def reset(self):
        self.actuators = self.initial_actuators

        if self.target_translation:
            self.target = np.array([uniform(-0.003, 0.003),
                                    uniform(-0.0015, 0.0015),
                                    uniform(0, 0.001),
                                    uniform(0, 0.001)])

        self.finished_steps = 0
        self.history = [{"objective": self.objective,
                         "reward": np.nan,
                         "observation": self.observation,
                         "action": np.full_like(self.action_space.high, np.nan)}]

        return self.observation

    def step(self, action):
        self.actuators += action
        reward = self.compute_reward()

        self.finished_steps += 1
        self.history.append({"objective": self.objective,
                             "reward": reward,
                             "observation": self.observation,
                             "action": action})
        
        done = all((np.abs(record["observation"][:4]) < self.goal).all() for record in self.history[-5:])

        return self.observation, reward, done, {}
    
    def render(self, mode="human", action_scale=1, observation_scale=1, reward_scale=1):
        fig = plt.figure("ARESEA", figsize=(14,12))
        fig.clear()

        gs = gridspec.GridSpec(3, 2, wspace=0.35, hspace=0.3, figure=fig)

        ax0 = fig.add_subplot(gs[0,:])
        self.plot_screen(ax0)
        
        ax1 = fig.add_subplot(gs[1,0])
        self.plot_actions(ax1, action_scale=action_scale)
        
        ax2 = fig.add_subplot(gs[1,1])
        self.plot_observations(ax2, observation_scale=observation_scale)

        ax3 = fig.add_subplot(gs[2,0])
        self.plot_rewards(ax3, reward_scale=reward_scale)
        
        ax4 = fig.add_subplot(gs[2,1])
        self.plot_objective(ax4)

        if mode == "rgb_array":
            return mplfig_to_npimage(fig)
        if mode == "human":
            plt.show()
        else:
            raise ValueError(f"Invalid render mode \"{mode}\" (allowed: {self.metadata['render.modes']})")
    
    def evaluate(self, actuators):
        """Evaluates the objective function for given actuator settings."""
        self.actuators = actuators

        self.history.append({"objective": self.objective,
                             "reward": np.nan,
                             "observation": self.observation,
                             "action": np.full_like(self.action_space.high, np.nan)})

        return self.objective
    
    @property
    def initial_actuators(self):
        """Return an action with the initial values for all actuators."""
        raise NotImplementedError
    
    @property
    def actuators(self):
        raise NotImplementedError
    
    @actuators.setter
    def actuators(self, values):
        raise NotImplementedError
    
    @property
    def screen_data(self):
        if self.magnets_changed:
            self._screen_data = self.read_screen()
            self.magnets_changed = False
        return self._screen_data

    def read_screen(self):
        """Get pixel data from the screen."""
        raise NotImplementedError
    
    @property
    def observation(self):
        return np.concatenate([self.beam_parameters, [self.beam_intensity], self.actuators])

    @property
    def beam_parameters(self):
        parameters = np.empty(4)
        for axis in [0, 1]:
            profile = self.screen_data.sum(axis=axis)
            minfiltered = minimum_filter1d(profile, size=5, mode="nearest")
            filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

            half_values, = np.where(filtered >= 0.5 * filtered.max())

            fwhm_pixel = half_values[-1] - half_values[0]
            center_pixel = half_values[0] + fwhm_pixel / 2

            parameters[axis] = (center_pixel - len(filtered) / 2) * self.pixel_size[axis] * self.binning
            parameters[axis+2] = fwhm_pixel / 2.355 * self.pixel_size[axis] * self.binning
            
        parameters[1] = -parameters[1]

        shifted = parameters - self.target

        return shifted
    
    @property
    def beam_intensity(self):
        return self.screen_data.sum()

    @property
    def objective(self):
        if self.beam_intensity < 0.88e5 and self.simulate_screen:
            intensity_component = 0.05
            return intensity_component
        else:
            if self.objective_method == "abs":
                parameter_component = np.sum([1,1,10,10] * np.abs(self.beam_parameters))
            elif self.objective_method == "square":
                parameter_component = np.square(self.beam_parameters).sum()
            elif self.objective_method == "max":
                parameter_component = max([1,1,10,10] * np.abs(self.beam_parameters))
            else:
                raise ValueError(f"Objective method \"{self.objective_method}\" is not available.")
            
            return parameter_component
            
    
    def compute_reward(self):
        if self.reward_method == "differential":
            previous_objective = self.history[-1]["objective"]
            return previous_objective - self.objective
        elif self.reward_method == "regret":
            return -self.objective
        else:
            raise ValueError(f"Reward method \"{self.reward_method}\" is not available.")

    def plot_screen(self, ax):
        ax.set_title("Screen")
        screen_extent = (-self.screen_resolution[0] * self.pixel_size[0] / 2, self.screen_resolution[0] * self.pixel_size[0] / 2,
                         -self.screen_resolution[1] * self.pixel_size[1] / 2, self.screen_resolution[1] * self.pixel_size[1] / 2)
        ax.imshow(self.screen_data, cmap="magma", interpolation="None", extent=screen_extent)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        target_mu_x, target_mu_y, target_sigma_x, target_sigma_y = self.target
        target_ellipse = Ellipse((target_mu_x,target_mu_y), target_sigma_x, target_sigma_y, fill=False, color="white")
        ax.add_patch(target_ellipse)

        read_mu_x, read_mu_y, read_sigma_x, read_sigma_y = self.target + self.beam_parameters
        read_ellipse = Ellipse((read_mu_x,read_mu_y), read_sigma_x, read_sigma_y, fill=False, color="deepskyblue", linestyle="--")
        ax.add_patch(read_ellipse)
    
    def plot_actions(self, ax, action_scale=1):
        actions = np.stack([record["action"] for record in self.history])
        actions = actions / action_scale

        ax.set_title("Actions")
        for i, name in enumerate(["$Q_1$", "$Q_2$", "$Q_3$", "$C_v$", "$C_h$"]):
            ax.plot(actions[:,i], label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (in Agent View)")
        ax.legend(loc="lower right")
        ax.grid()
    
    def plot_observations(self, ax, observation_scale=1):
        observations = np.stack([record["observation"] for record in self.history])
        observations = observations / observation_scale

        ax.set_title("Observations")
        for i, name in enumerate(["$\mu_x$", "$\mu_y$", "$\sigma_x$", "$\sigma_y$", "Intensity",
                                  "$Q_1$", "$Q_2$", "$Q_3$", "$C_v$", "$C_h$"]):
            ax.plot(observations[:,i], label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (in Agent View)")
        ax.legend(loc="upper right")
        ax.grid()
    
    def plot_rewards(self, ax, reward_scale=1):
        rewards = np.array([record["reward"] for record in self.history])
        rewards = rewards / reward_scale

        ax.set_title("Reward")
        ax.plot(rewards, label="Reward")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.grid()

        axt = ax.twinx()
        axt.plot([], label="Reward")
        cumulative = [sum(reward for reward in np.insert(rewards[1:i+1], 0, 0))
                                 for i in range(len(rewards))]
        axt.plot(cumulative, label="Cumulative Reward")
        axt.set_ylabel("Cumulative Reward")
        axt.legend(loc="upper right")
    
    def plot_objective(self, ax):
        ax.set_title("Objective Function")
        ax.plot([record["objective"] for record in self.history], label="Objective Function")
        ax.set_xlabel("Step")
        ax.set_ylabel("Objective Function")
        ax.grid()
