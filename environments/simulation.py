import gym
from gym import spaces
import cheetah
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np

from . import ARESlatticeStage3v1_9 as lattice
from . import utils


class ARESEACheetah(gym.Env):
    """ARESEA version using a Cheetah simulation as its backend."""

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 5
    }

    accelerator_observation_space = spaces.Dict({
        "observation": spaces.Box(
            low=np.array([0, -30, -30, -30, -3e-3, -6e-3], dtype=np.float32),
            high=np.array([1e5, 30, 30, 30, 3e-3, 6e-3], dtype=np.float32)
        ),
        "desired_goal": spaces.Box(
            low=np.array([-2e-3, -2e-3, 0, 0], dtype=np.float32),
            high=np.array([2e-3, 2e-3, 5e-4, 5e-4], dtype=np.float32)
        ),
        "achieved_goal": spaces.Box(
            low=np.array([-2e-3, -2e-3, 0, 0], dtype=np.float32),
            high=np.array([2e-3,  2e-3, 5e-4, 5e-4], dtype=np.float32)
        )
    })
    accelerator_action_space = spaces.Box(
        low=accelerator_observation_space["observation"].low[-5:] * 0.1,
        high=accelerator_observation_space["observation"].high[-5:] * 0.1
    )
    accelerator_optimization_space = spaces.Box(
        low=accelerator_observation_space["observation"].low[-5:],
        high=accelerator_observation_space["observation"].high[-5:]
    )
    accelerator_reward_range = (
        -accelerator_observation_space["achieved_goal"].high[0] * 250,
        accelerator_observation_space["achieved_goal"].high[0] * 250
    )
    
    binning = 4
    screen_resolution = (2448, 2040)
    pixel_size = (3.3198e-6, 2.4469e-6)

    def __init__(self):
        cell = utils.subcell_of(lattice.cell, "AREASOLA1", "AREABSCR1")
        self.segment = cheetah.Segment.from_ocelot(cell)
        self.segment.AREABSCR1.resolution = self.screen_resolution
        self.segment.AREABSCR1.pixel_size = self.pixel_size
        self.segment.AREABSCR1.binning = self.binning
        self.segment.AREABSCR1.is_active = True

        self.magnets_changed = True
    
    def reset(self):
        self.incoming = cheetah.Beam.make_random(
            n=int(1e5),
            mu_x=np.random.uniform(-5e-4, 5e-4),
            mu_y=np.random.uniform(-5e-4, 5e-4),
            mu_xp=np.random.uniform(-1e-4, 1e-4),
            mu_yp=np.random.uniform(-1e-4, 1e-4),
            sigma_x=np.random.uniform(0, 5e-4),
            sigma_y=np.random.uniform(0, 5e-4),
            sigma_xp=np.random.uniform(0, 1e-4),
            sigma_yp=np.random.uniform(0, 1e-4),
            sigma_s=np.random.uniform(0, 1e-4),
            sigma_p=np.random.uniform(0, 1e-3)
        )

        self.actuators = self.initial_actuators
        
        self.goal = self.accelerator_observation_space["desired_goal"].sample()

        self.finished_steps = 0
        objective = self.compute_objective(
            self.observation["achieved_goal"],
            self.observation["desired_goal"]
        )
        self.history = [{
            "objective": objective,
            "reward": np.nan,
            "observation": self.observation,
            "action": np.full_like(self.action_space.high, np.nan)
        }]

        return self.observation2agent(self.observation)
    
    def step(self, action):
        action = self.action2accelerator(action)

        self.actuators += action

        info = {"previous_objective": self.history[-1]["objective"]}

        reward = self.compute_reward(
            self.observation["achieved_goal"],
            self.observation["desired_goal"],
            info
        )
        objective = self.compute_objective(
            self.observation["achieved_goal"],
            self.observation["desired_goal"]
        )

        self.finished_steps += 1
        self.history.append({
            "objective": objective,
            "reward": reward,
            "observation": self.observation,
            "action": action
        })

        done = all(abs(achieved - desired) < 5e-6 for achieved, desired in zip(self.observation["achieved_goal"], self.observation["desired_goal"]))

        return self.observation2agent(self.observation), self.reward2agent(reward), done, info
    
    def render(self, mode="human"):
        fig = plt.figure("ARESEA-Cheetah", figsize=(28,8))
        fig.clear()

        gs = gridspec.GridSpec(2, 4, wspace=0.35, hspace=0.3, figure=fig)

        ax_screen = fig.add_subplot(gs[0,0])
        self.plot_screen(ax_screen)
        
        sgs_trace = gridspec.GridSpecFromSubplotSpec(3, 1, hspace=0, height_ratios=[2,2,1], subplot_spec=gs[1,0])
        ax_tracex = fig.add_subplot(sgs_trace[0,0])
        ax_tracey = fig.add_subplot(sgs_trace[1,0])
        ax_lat = fig.add_subplot(sgs_trace[2,0])
        self.plot_beam_overview(ax_tracex, ax_tracey, ax_lat)

        ax_obs = fig.add_subplot(gs[0,1])
        self.plot_observations(ax_obs)

        ax_goal = fig.add_subplot(gs[1,1])
        self.plot_goals(ax_goal)
        
        sgs_act = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[1,2,1], subplot_spec=gs[:,2])
        ax_act = fig.add_subplot(sgs_act[1,0])
        self.plot_actions(ax_act)

        ax_rew = fig.add_subplot(gs[0,3])
        self.plot_rewards(ax_rew)
        
        ax_obj = fig.add_subplot(gs[1,3])
        self.plot_objective(ax_obj)

        if mode == "rgb_array":
            return mplfig_to_npimage(fig)
        if mode == "human":
            plt.show()
        else:
            raise ValueError(f"Invalid render mode \"{mode}\" (allowed: {self.metadata['render.modes']})")
    
    @property
    def initial_actuators(self):
        return self.accelerator_optimization_space.sample()

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
        
        self.screen_data = self.read_screen()
        
    def read_screen(self):
        self.run_simulation()
        return self.segment.AREABSCR1.reading
    
    def run_simulation(self):
        _ = self.segment(self.incoming)
    
    @property
    def beam_parameters(self):
        return np.array([
            self.segment.AREABSCR1.read_beam.mu_x,
            self.segment.AREABSCR1.read_beam.mu_y,
            self.segment.AREABSCR1.read_beam.sigma_x,
            self.segment.AREABSCR1.read_beam.sigma_y
        ])

    @property
    def observation(self):
        return {
            "observation": np.concatenate([[self.beam_intensity], self.actuators]),
            "desired_goal": self.goal,
            "achieved_goal": self.beam_parameters
        }

    @property
    def beam_intensity(self):
        return self.screen_data.sum()
    
    def compute_objective(self, achieved_goal, desired_goal):
        offset = achieved_goal - desired_goal
        weights = np.array([1, 1, 2, 2])

        # Weighted sum of absolute beam parameters
        shaped = np.log((weights * np.abs(offset)).sum())
        boni = sum(10 for achieved, desired in zip(achieved_goal, desired_goal) if abs(achieved - desired) < 5e-6)
        win = 100 * all(abs(achieved - desired) < 5e-6 for achieved, desired in zip(achieved_goal, desired_goal))
        
        return shaped - boni - win

        # Maximum of absolute beam parameters
        # return (weights * np.abs(offset)).max()

    def compute_reward(self, achieved_goal, desired_goal, info):
        current_objective = self.compute_objective(achieved_goal, desired_goal)
        previous_objective = info["previous_objective"]

        reward = previous_objective - current_objective

        return reward if reward > 0 else 2 * reward
    
    def evaluate(self, actuators):
        """Evaluates the objective function for given actuator settings."""
        self.actuators = actuators
        objective = self.compute_objective(self.beam_parameters, self.goal)

        self.history.append({"objective": objective,
                             "reward": np.nan,
                             "observation": self.observation,
                             "action": np.full_like(self.action_space.high, np.nan)})

        return self.objective
    
    @property
    def observation_space(self):
        return spaces.Dict({
            k: spaces.Box(
                low=self.accelerator_observation_space[k].low / self.accelerator_observation_space[k].high,
                high=self.accelerator_observation_space[k].high / self.accelerator_observation_space[k].high
            ) for k in self.accelerator_observation_space.spaces.keys()
        })
    
    @property
    def action_space(self):
        return spaces.Box(
            low=self.accelerator_action_space.low / self.accelerator_action_space.high,
            high=self.accelerator_action_space.high / self.accelerator_action_space.high
        )
    
    @property
    def reward_range(self):
        return (
            self.accelerator_reward_range[0] / self.accelerator_reward_range[1],
            self.accelerator_reward_range[1] / self.accelerator_reward_range[1]
        )
    
    def observation2agent(self, observation):
        """Convert an observation from accelerator view to agent view."""
        return {k: observation[k] / self.accelerator_observation_space[k].high for k in observation.keys()}
    
    def action2accelerator(self, action):
        """Convert an action from agent view to accelerator view."""
        return action * self.accelerator_action_space.high
    
    def action2agent(self, action):
        """Convert an action from accelerator view to agent view."""
        return action / self.accelerator_action_space.high
    
    def reward2agent(self, reward):
        """Convert a reward from accelerator view to agent view."""
        return reward / self.accelerator_reward_range[1]
    
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

        target_mu_x, target_mu_y, target_sigma_x, target_sigma_y = self.goal
        target_ellipse = Ellipse((target_mu_x,target_mu_y), target_sigma_x, target_sigma_y, fill=False, color="white")
        ax.add_patch(target_ellipse)

        read_mu_x, read_mu_y, read_sigma_x, read_sigma_y = self.beam_parameters
        read_ellipse = Ellipse((read_mu_x,read_mu_y), read_sigma_x, read_sigma_y, fill=False, color="deepskyblue", linestyle="--")
        ax.add_patch(read_ellipse)

        true_mu_x, true_mu_y, true_sigma_x, true_sigma_y = [self.segment.AREABSCR1.read_beam.mu_x,
                                                            self.segment.AREABSCR1.read_beam.mu_y,
                                                            self.segment.AREABSCR1.read_beam.sigma_x,
                                                            self.segment.AREABSCR1.read_beam.sigma_y]
        true_ellipse = Ellipse((true_mu_x,true_mu_y), true_sigma_x, true_sigma_y, fill=False, color="lime", linestyle="--")
        ax.add_patch(true_ellipse)
    
    def plot_actions(self, ax):
        actions = np.stack([self.action2agent(record["action"]) for record in self.history])

        ax.set_title("Actions")
        for i, name in enumerate([r"$\Delta k_{Q_1}$", r"$\Delta k_{Q_2}$", r"$\Delta k_{Q_3}$",
                                  r"$\Delta \alpha_{C_v}$", r"$\Delta \alpha_{C_h}$"]):
            ax.plot(actions[:,i], label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (in Agent View)")
        ax.legend(loc="upper right", bbox_to_anchor=(1,-0.18), ncol=3)
        ax.grid()
    
    def plot_observations(self, ax):
        observations = np.stack([self.observation2agent(record["observation"])["observation"]
                                     for record in self.history])

        ax.set_title("Observations")
        for i, name in enumerate([r"$i_S$", r"$k_{Q_1}$", r"$k_{Q_2}$", r"$k_{Q_3}$",
                                  r"$\alpha_{C_v}$", r"$\alpha_{C_h}$"]):
            ax.plot(observations[:,i], label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (in Agent View)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01,1), ncol=2)
        ax.grid()
    
    def plot_goals(self, ax):
        desired_goals = np.stack([self.observation2agent(record["observation"])["desired_goal"]
                                      for record in self.history])

        achieved_goals = np.stack([self.observation2agent(record["observation"])["achieved_goal"]
                                       for record in self.history])
        
        ax.set_title("Goals")
        for i, name in enumerate([r"$\mu_x$", r"$\mu_y$", r"$\sigma_x$", r"$\sigma_y$"]):
            p, = ax.plot(achieved_goals[:,i], label=name)
            ax.plot(desired_goals[:,i], label=f"{name}'", color=p.get_color(), ls="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (in Agent View)")
        ax.legend(loc="lower left", bbox_to_anchor=(1.01,0), ncol=2)
        ax.grid()
    
    def plot_rewards(self, ax):
        rewards = np.array([self.reward2agent(record["reward"]) for record in self.history])

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
        axt.legend(loc="upper right", bbox_to_anchor=(-0.12,1))
    
    def plot_objective(self, ax):
        ax.set_title("Objective Function")
        ax.plot([record["objective"] for record in self.history], label="Objective Function")
        ax.set_xlabel("Step")
        ax.set_ylabel("Objective Function")
        ax.grid()
