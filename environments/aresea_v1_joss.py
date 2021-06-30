import joss
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
from numpy.random import uniform

from accelerator_environments import utils
import accelerator_environments.envs.ares.ARESlatticeStage3v1_9 as lattice
from accelerator_environments.envs.ares.aresea_v1 import ARESEA


class ARESEAJOSS(ARESEA):
    """ARESEA version using a JOSS simulation as its backend."""

    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 5}

    def __init__(self, random_incoming=True, random_actuators=True, simulate_screen=True, **kwargs):
        super().__init__(**kwargs)

        self.random_incoming = random_incoming
        self.random_actuators = random_actuators
        self.simulate_screen = simulate_screen

        cell = utils.subcell_of(lattice.cell, "AREASOLA1", "ARMRBSCR1")
        self.segment = joss.Segment.from_ocelot(cell)
        self.segment.ARMRBSCR1.resolution = self.screen_resolution
        self.segment.ARMRBSCR1.pixel_size = self.pixel_size
        self.segment.ARMRBSCR1.binning = self.binning
        self.segment.ARMRBSCR1.is_active = True
    
    def reset(self):
        self.actuators = self.initial_actuators
        if self.random_incoming:
            self.incoming = joss.Beam.make_random(n=int(1e5),
                                                  mu_x=uniform(-5e-4, 5e-4),
                                                  mu_y=uniform(-5e-4, 5e-4),
                                                  mu_xp=uniform(-1e-4, 1e-4),
                                                  mu_yp=uniform(-1e-4, 1e-4),
                                                  sigma_x=uniform(0, 5e-4),
                                                  sigma_y=uniform(0, 5e-4),
                                                  sigma_xp=uniform(0, 1e-4),
                                                  sigma_yp=uniform(0, 1e-4),
                                                  sigma_s=uniform(0, 1e-4),
                                                  sigma_p=uniform(0, 1e-3))
        else:
            self.incoming = joss.Beam.make_random(n=int(1e+5), sigma_x=175e-6, sigma_y=175e-6)
        
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
    
    @property
    def initial_actuators(self):
        if self.random_actuators:
            return self.optimization_space.sample()
        else:
            return np.zeros(self.optimization_space.shape)
        
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
        
    def read_screen(self):
        _ = self.segment(self.incoming)
        image = self.segment.ARMRBSCR1.reading

        return image
    
    @property
    def beam_parameters(self):
        if self.simulate_screen:
            return super().beam_parameters
        else:
            _ = self.screen_data    # TODO: This is a hack for now
            parameters = np.array([self.segment.ARMRBSCR1.read_beam.mu_x,
                                   self.segment.ARMRBSCR1.read_beam.mu_y,
                                   self.segment.ARMRBSCR1.read_beam.sigma_x,
                                   self.segment.ARMRBSCR1.read_beam.sigma_y])
            
            shifted = parameters - self.target

            return shifted
    
    def plot_beam_overview(self, axx, axy, axl):
        axx.set_title(f"Beam Overview in Step {self.finished_steps}")
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

        true_mu_x, true_mu_y, true_sigma_x, true_sigma_y = [self.segment.ARMRBSCR1.read_beam.mu_x,
                                                            self.segment.ARMRBSCR1.read_beam.mu_y,
                                                            self.segment.ARMRBSCR1.read_beam.sigma_x,
                                                            self.segment.ARMRBSCR1.read_beam.sigma_y]
        true_ellipse = Ellipse((true_mu_x,true_mu_y), true_sigma_x, true_sigma_y, fill=False, color="lime", linestyle="--")
        ax.add_patch(true_ellipse)
    
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
