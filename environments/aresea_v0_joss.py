import joss
import matplotlib.pyplot as plt
from matplotlib import gridspec
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np

from accelerator_environments import utils
import accelerator_environments.envs.ares.ARESlatticeStage3v1_9 as lattice
from accelerator_environments.envs.ares.aresea_v0 import ARESEA


class ARESEAJOSS(ARESEA):
    """Ocelot version of the ARES experimental area."""

    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 5}

    def __init__(self):
        super().__init__()

        cell = utils.subcell_of(lattice.cell, "AREASOLA1", "ARMRBSCR1")
        self.segment = joss.Segment(cell)
        self.segment.ARMRBSCR1.is_active = True

        self.incoming_particles = joss.random_particles(n=int(1e+5),
                                                        sigma_x=175e-6,
                                                        sigma_y=175e-6)

        self.screen_bin_edges = (np.linspace(-self.screen_resolution[0] * self.pixel_size[0] / 2,
                                             self.screen_resolution[0] * self.pixel_size[0] / 2,
                                             num=int(self.screen_resolution[0] / self.binning) + 1),
                                 np.linspace(-self.screen_resolution[1] * self.pixel_size[1] / 2,
                                             self.screen_resolution[1] * self.pixel_size[1] / 2,
                                             num=int(self.screen_resolution[1] / self.binning) + 1))
        
    def initial_actuator_values(self):
        """Return an action with the initial values for all actuators."""
        return np.zeros_like(self.optimization_space.high)
        
    def read_magnets(self):
        return np.array([self.segment.AREAMQZM1.k1,
                         self.segment.AREAMQZM2.k1,
                         self.segment.AREAMQZM3.k1,
                         self.segment.AREAMCVM1.angle,
                         self.segment.AREAMCHM1.angle])

    def write_magnets(self, values):
        """Set magents to the values in the given action."""
        self.segment.AREAMQZM1.k1, self.segment.AREAMQZM2.k1, self.segment.AREAMQZM3.k1 = values[:3]
        self.segment.AREAMCVM1.angle, self.segment.AREAMCHM1.angle = values[3:]
        
    def read_screen(self):
        """Get pixel data from the screen."""
        self.particles = self.segment(self.incoming_particles)
        image, _, _ = np.histogram2d(self.particles[:,0], self.particles[:,2],
                                     bins=self.screen_bin_edges)
        image = np.flipud(image.T)

        return image
    
    def beam_parameters(self):
        self.screen_data = self.read_screen()
        parameters = np.array([self.particles[:,0].mean(),
                               self.particles[:,2].mean(),
                               self.particles[:,0].std(),
                               self.particles[:,2].std()])
        return parameters / self.beam_parameter_scalars
    
    def render(self, mode="rgb_array", close=False):
        fig = plt.figure("ARESEA-JOSS", figsize=(14,12))
        fig.clear()

        gs = gridspec.GridSpec(3, 2, wspace=0.35, hspace=0.3, figure=fig)
        sgs0 = gridspec.GridSpecFromSubplotSpec(3, 1, hspace=0, height_ratios=[2,2,1], subplot_spec=gs[0,0])

        ax0 = fig.add_subplot(sgs0[0,0])
        ax1 = fig.add_subplot(sgs0[1,0])
        ax2 = fig.add_subplot(sgs0[2,0])
        ax0.set_title(f"Beam Overview in Step {self.finished_steps}")
        self.segment.plot_reference_particle_traces(ax0, ax1, particles=self.incoming_particles)
        self.segment.plot(ax2, s=0)

        ax3 = fig.add_subplot(gs[0,1])
        ax3.set_title("Screen")
        screen_extent = (-self.screen_resolution[0] * self.pixel_size[0] / 2, self.screen_resolution[0] * self.pixel_size[0] / 2,
                         -self.screen_resolution[1] * self.pixel_size[1] / 2, self.screen_resolution[1] * self.pixel_size[1] / 2)
        ax3.imshow(self.screen_data, cmap="magma", interpolation="None", extent=screen_extent)
        ax3.set_xlabel("x (m)")
        ax3.set_ylabel("y (m)")

        ax4 = fig.add_subplot(gs[1,0])
        ax4.set_title("Actions")
        for i, name in enumerate(["Q1", "Q2", "Q3", "CV", "CH"]):
            ax4.plot([record["action"][i] for record in self.history], label=name)
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Value (in Agent View)")
        ax4.legend(loc="lower right")
        ax4.grid()
        
        ax5 = fig.add_subplot(gs[1,1])
        ax5.set_title("Observations")
        for i, name in enumerate(["x", "y", "w", "h", "Q1", "Q2", "Q3", "CV", "CH"]):
            ax5.plot([record["observation"][i] for record in self.history], label=name)
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Value (in Agent View)")
        ax5.legend(loc="upper right")
        ax5.grid()

        ax6 = fig.add_subplot(gs[2,0])
        ax6.set_title("Reward")
        ax6.plot([record["reward"] for record in self.history], label="Reward")
        ax6.set_xlabel("Step")
        ax6.set_ylabel("Reward")
        ax6.grid()
        ax7 = ax6.twinx()
        ax7.plot([], label="Reward")
        cumulative = [sum([record["reward"] for record in self.history][:i+1])
                                            for i in range(len(self.history))]
        ax7.plot(cumulative, label="Cumulative Reward")
        ax7.set_ylabel("Cumulative Reward")
        ax7.legend(loc="upper right")
        
        ax8 = fig.add_subplot(gs[2,1])
        ax8.set_title("Objective Function")
        ax8.plot([record["score"] for record in self.history], label="Objective Function")
        ax8.set_xlabel("Step")
        ax8.set_ylabel("Objective Function")
        ax8.grid()

        if mode == "rgb_array":
            return mplfig_to_npimage(fig)
        if mode == "human":
            plt.show()
        else:
            raise ValueError(f"Invalid render mode \"{mode}\" (allowed: {self.metadata['render.modes']})")
