from accelerator_environments import utils
from gym import Env
from gym.spaces import Box
import joss
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np

from . import ARESlatticeStage3v1_9 as lattice


class ARESEAJOSS(Env):
    """ARESEA version using a JOSS simulation as its backend."""

    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 5}

    observation_space = Box(low=np.array([-4e-3, -4e-3,    0,    0,   0, -30, -30, -30, -3e-3, -3e-3], dtype=np.float32),
                            high=np.array([4e-3,  4e-3, 4e-4, 4e-4, 1e5,  30,  30,  30,  3e-3,  3e-3], dtype=np.float32))
    action_space = Box(low=observation_space.low[-5:] * 0.1,
                       high=observation_space.high[-5:] * 0.1)
    optimization_space = Box(low=observation_space.low[-5:],
                             high=observation_space.high[-5:])
    
    binning = 4
    screen_resolution = (2448, 2040)
    pixel_size = (3.3198e-6, 2.4469e-6)

    target = np.array([0, 0, 0, 0])
    goal = np.array([1e-4, 1e-4, 5e-4, 5e-4])

    def __init__(self):
        super().__init__()

        cell = utils.subcell_of(lattice.cell, "AREASOLA1", "AREABSCR1")
        self.segment = joss.Segment.from_ocelot(cell)
        self.segment.AREABSCR1.resolution = self.screen_resolution
        self.segment.AREABSCR1.pixel_size = self.pixel_size
        self.segment.AREABSCR1.binning = self.binning
        self.segment.AREABSCR1.is_active = True

        self.magnets_changed = True
    
    def reset(self):
        self.actuators = self.initial_actuators

        self.incoming = joss.Beam.make_random(n=int(1e5),
                                              mu_x=np.random.uniform(-5e-4, 5e-4),
                                              mu_y=np.random.uniform(-5e-4, 5e-4),
                                              mu_xp=np.random.uniform(-1e-4, 1e-4),
                                              mu_yp=np.random.uniform(-1e-4, 1e-4),
                                              sigma_x=np.random.uniform(0, 5e-4),
                                              sigma_y=np.random.uniform(0, 5e-4),
                                              sigma_xp=np.random.uniform(0, 1e-4),
                                              sigma_yp=np.random.uniform(0, 1e-4),
                                              sigma_s=np.random.uniform(0, 1e-4),
                                              sigma_p=np.random.uniform(0, 1e-3))
        
        self.target = np.array([np.random.uniform(-0.003, 0.003),
                                np.random.uniform(-0.0015, 0.0015),
                                np.random.uniform(0, 0.001),
                                np.random.uniform(0, 0.001)])

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
        fig = plt.figure("ARESEA-JOSS", figsize=(14,12))
        fig.clear()

        gs = gridspec.GridSpec(3, 2, wspace=0.35, hspace=0.3, figure=fig)
        sgs0 = gridspec.GridSpecFromSubplotSpec(3, 1, hspace=0, height_ratios=[2,2,1], subplot_spec=gs[0,0])

        ax0 = fig.add_subplot(sgs0[0,0])
        ax1 = fig.add_subplot(sgs0[1,0])
        ax2 = fig.add_subplot(sgs0[2,0])
        self.plot_beam_overview(ax0, ax1, ax2)

        ax3 = fig.add_subplot(gs[0,1])
        self.plot_screen(ax3)
        
        ax4 = fig.add_subplot(gs[1,0])
        self.plot_actions(ax4, action_scale=action_scale)
        
        ax5 = fig.add_subplot(gs[1,1])
        self.plot_observations(ax5, observation_scale=observation_scale)

        ax6 = fig.add_subplot(gs[2,0])
        self.plot_rewards(ax6, reward_scale=reward_scale)
        
        ax7 = fig.add_subplot(gs[2,1])
        self.plot_objective(ax7)

        if mode == "rgb_array":
            return mplfig_to_npimage(fig)
        if mode == "human":
            plt.show()
        else:
            raise ValueError(f"Invalid render mode \"{mode}\" (allowed: {self.metadata['render.modes']})")
    
    @property
    def actuators(self):
        return np.array([self.segment.AREAMQZM1.k1,
                         self.segment.AREAMQZM2.k1,
                         self.segment.AREAMQZM3.k1,
                         self.segment.AREAMCVM1.angle,
                         self.segment.AREAMCHM1.angle])

    @actuators.setter
    def actuators(self, values):
        self.segment.AREAMQZM1.k1, self.segment.AREAMQZM2.k1, self.segment.AREAMQZM3.k1 = values[:3]
        self.segment.AREAMCVM1.angle, self.segment.AREAMCHM1.angle = values[3:]
        
        self.magnets_changed = True
    
    @property
    def initial_actuators(self):
        return self.optimization_space.sample()
    
    @property
    def screen_data(self):
        if self.magnets_changed:
            self._screen_data = self.read_screen()
            self.magnets_changed = False
        return self._screen_data
        
    def read_screen(self):
        _ = self.segment(self.incoming)
        image = self.segment.AREABSCR1.reading

        return image
    
    @property
    def beam_parameters(self):
        _ = self.screen_data    # TODO: This is a hack for now
        parameters = np.array([self.segment.AREABSCR1.read_beam.mu_x,
                                self.segment.AREABSCR1.read_beam.mu_y,
                                self.segment.AREABSCR1.read_beam.sigma_x,
                                self.segment.AREABSCR1.read_beam.sigma_y]) 
        
        shifted = parameters - self.target

        return shifted

    @property
    def observation(self):
        return np.concatenate([self.beam_parameters, [self.beam_intensity], self.actuators])
    
    @property
    def beam_intensity(self):
        return self.screen_data.sum()
    
    @property
    def objective(self):
        # Weighted sum of absolute beam parameters
        objective = np.sum([1,1,10,10] * np.abs(self.beam_parameters))

        # Maximum of absolute beam parameters
        # objective = np.max([1,1,10,10] * np.abs(self.beam_parameters))

        return objective
    
    def compute_reward(self):
        # Differential
        previous_objective = self.history[-1]["objective"]
        reward = previous_objective - self.objective
        
        # Regret
        # reward = -self.objective

        return reward
    
    def evaluate(self, actuators):
        """Evaluates the objective function for given actuator settings."""
        self.actuators = actuators

        self.history.append({"objective": self.objective,
                             "reward": np.nan,
                             "observation": self.observation,
                             "action": np.full_like(self.action_space.high, np.nan)})

        return self.objective
    
    def plot_beam_overview(self, axx, axy, axl):
        axx.set_title(f"Beam Overview")
        self.segment.plot_reference_particle_traces(axx, axy, beam=self.incoming)
        self.segment.plot(axl, s=0)
    
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

        true_mu_x, true_mu_y, true_sigma_x, true_sigma_y = [self.segment.AREABSCR1.read_beam.mu_x,
                                                            self.segment.AREABSCR1.read_beam.mu_y,
                                                            self.segment.AREABSCR1.read_beam.sigma_x,
                                                            self.segment.AREABSCR1.read_beam.sigma_y]
        true_ellipse = Ellipse((true_mu_x,true_mu_y), true_sigma_x, true_sigma_y, fill=False, color="lime", linestyle="--")
        ax.add_patch(true_ellipse)
    
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
