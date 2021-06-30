from gym.spaces import Box
import numpy as np

import pickle

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
from moviepy.video.io.bindings import mplfig_to_npimage

import joss

from numpy.random import uniform

from accelerator_environments.envs.ares.aresea_v1_joss import ARESEAJOSS


class ARESEA2D(ARESEAJOSS):
    
    def __init__(self, random_incoming, random_actuators, **kwargs):
        super().__init__(random_incoming=random_incoming, random_actuators=random_actuators, **kwargs)

        self.observation_space = Box(low=self.observation_space.low[:7],
                                     high=self.observation_space.high[:7])
        self.action_space = Box(low=self.action_space.low[:2],
                                high=self.action_space.high[:2])
        self.optimization_space = Box(low=self.optimization_space.low[:2],
                                      high=self.optimization_space.high[:2])


        steps = 1001
        with open(f'data_{steps}.pkl','rb') as f:
            self.d = pickle.load(f)
        with open(f'ks_{steps}.pkl','rb') as f:
            self.ks = pickle.load(f)

        self.d = np.rot90(self.d)


    @property
    def actuators(self):
        return np.array([self.segment.AREAMQZM1.k1,
                         self.segment.AREAMQZM2.k1])

    @actuators.setter
    def actuators(self, values):
        self.segment.AREAMQZM1.k1, self.segment.AREAMQZM2.k1 = values
        
        self.magnets_changed = True

    def plot_actions(self, ax, action_scale=1):
        actions = np.stack([record["action"] for record in self.history])
        actions = actions / action_scale

        ax.set_title("Actions")
        for i, name in enumerate(["Q1", "Q2"]):
            ax.plot(actions[:,i], label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (in Agent View)")
        ax.legend(loc="lower right")
        ax.grid()

    def plot_observations(self, ax, observation_scale=1):
        observations = np.stack([record["observation"] for record in self.history])
        observations = observations / observation_scale

        ax.set_title("Observations")
        for i, name in enumerate(["x", "y", "w", "h", "Q1", "Q2"]):
            ax.plot(observations[:,i], label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (in Agent View)")
        ax.legend(loc="upper right")
        ax.grid()
    
    def plot_actuators(self, ax, actuator_scale=1):
        actuators = np.stack([record["observation"][-2:] for record in self.history])
        actuators = actuators / actuator_scale

        X, Y = np.meshgrid(self.ks, self.ks)

        levels = [0.001,0.02,0.05,0.1,0.2,0.3,.4]
        CS = ax.contour(self.d, levels,
                        origin='lower',
                    #  cmap='magma',
                        colors = 'k',
                        linewidths=1, 
                        extent=[self.ks[0],self.ks[-1],self.ks[-1],self.ks[0]])
        ax.clabel(CS, fontsize=9, inline=True)
        im = ax.imshow(self.d, interpolation='bilinear', 
                    #    origin='lower',
                    cmap=cm.plasma, extent=[self.ks[0],self.ks[-1],self.ks[0],self.ks[-1]])

        CBI = plt.colorbar(im, orientation='horizontal', shrink=0.5)

        ax.set_aspect('equal')
        ax.set_title("Actuators")
        ax.plot(actuators[:,0],actuators[:,1],color='red', marker='x', label='Trajectory')
        ax.set_xlabel("Q1")
        ax.set_ylabel("Q2")
        # ax.set_xlim(-10,10)
        # ax.set_ylim(-10,10)
        ax.legend(loc="lower right")
        ax.grid()
    
    def render(self, mode="human", action_scale=1, observation_scale=1, reward_scale=1):
        fig = plt.figure("ARESEA-JOSS", figsize=(24,15))
        fig.clear()

        gs = gridspec.GridSpec(3, 4, wspace=0.35, hspace=0.3, figure=fig)
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

        ax8 = fig.add_subplot(gs[:,2:])
        self.plot_actuators(ax8,actuator_scale=1)

        if mode == "rgb_array":
            return mplfig_to_npimage(fig)
        if mode == "human":
            plt.show()
        else:
            raise ValueError(f"Invalid render mode \"{mode}\" (allowed: {self.metadata['render.modes']})")

