from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import gridspec
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import ocelot as oc

from . import ARESlatticeStage3v1_9 as ares
from . import simulation
from . import utils


class ARESEAOcelot(simulation.ARESEACheetah):
    """
    Ocelot version of the ARES EA environment that adds the simulation of second order dynamics and
    space charge effects.
    """

    def __init__(self):
        self.cell = utils.subcell_of(ares.cell, "AREASOLA1", "AREABSCR1")
        self.method = oc.MethodTM()
        self.method.global_method = oc.SecondTM
        self.lattice = oc.MagneticLattice(self.cell, method=self.method)

        self.screen_bin_edges = (np.linspace(-self.screen_resolution[0] * self.pixel_size[0] / 2,
                                             self.screen_resolution[0] * self.pixel_size[0] / 2,
                                             num=int(self.screen_resolution[0] / self.binning) + 1),
                                 np.linspace(-self.screen_resolution[1] * self.pixel_size[1] / 2,
                                             self.screen_resolution[1] * self.pixel_size[1] / 2,
                                             num=int(self.screen_resolution[1] / self.binning) + 1))

        self.magnets_changed = True
    
    def reset(self):
        self.incoming = oc.generate_parray(
            nparticles=int(1e+5),
            sigma_x=np.random.uniform(0, 5e-4),
            sigma_y=np.random.uniform(0, 5e-4),
            sigma_px=np.random.uniform(0, 1e-4),
            sigma_py=np.random.uniform(0, 1e-4),
            sigma_p=np.random.uniform(0, 1e-3),
            energy=0.1
        )
        mu_x = np.random.uniform(-5e-4, 5e-4)
        mu_y = np.random.uniform(-5e-4, 5e-4)
        mu_xp = np.random.uniform(-1e-4, 1e-4)
        mu_yp = np.random.uniform(-1e-4, 1e-4)
        self.incoming.rparticles[0,:] += mu_x
        self.incoming.rparticles[1,:] += mu_xp
        self.incoming.rparticles[2,:] += mu_y
        self.incoming.rparticles[3,:] += mu_yp
        # import ocelot.adaptors.astra2ocelot as oca
        # self.incoming = oca.astraBeam2particleArray("environments/ACHIP_EA1_2021.1351.001", print_params=False)

        self.actuators = self.initial_actuators
        
        self.goal = self.accelerator_observation_space["desired_goal"].sample()
        # self.goal = np.array([-0.000808033, -0.0013411774, 0.0, 0.0])

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
    
    def render(self, mode="human"):
        fig = plt.figure("ARESEA-Ocelot", figsize=(28,8))
        fig.clear()

        gs = gridspec.GridSpec(2, 4, wspace=0.35, hspace=0.3, figure=fig)

        ax_screen = fig.add_subplot(gs[0,0])
        self.plot_screen(ax_screen)

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
    def actuators(self):
        return np.array([
            ares.areamqzm1.k1,
            ares.areamqzm2.k1,
            ares.areamqzm3.k1,
            ares.areamcvm1.angle,
            ares.areamchm1.angle
        ])
    
    @actuators.setter
    def actuators(self, values):
        ares.areamqzm1.k1, ares.areamqzm2.k1, ares.areamqzm3.k1 = values[:3]
        ares.areamcvm1.angle, ares.areamchm1.angle = values[3:]
        
        self.screen_data = self.read_screen()

    def read_screen(self):
        self.run_simulation()
        
        image, _, _ = np.histogram2d(self.outgoing.rparticles[0,:],
                                     self.outgoing.rparticles[2,:],
                                     bins=self.screen_bin_edges)
        image = np.flipud(image.T)
                
        return image

    def run_simulation(self):
        self.lattice.update_transfer_maps()
        navigator = oc.Navigator(self.lattice)
        navigator.unit_step = 0.1
        sc = oc.SpaceCharge()
        navigator.add_physics_proc(sc, self.lattice.sequence[0], self.lattice.sequence[-1])

        _, self.outgoing = oc.track(self.lattice, deepcopy(self.incoming), navigator,
                                    calc_tws=False, print_progress=False)
    
    @property
    def beam_parameters(self):
        return np.array([
            self.outgoing.x().mean(),
            self.outgoing.y().mean(),
            self.outgoing.x().std(),
            self.outgoing.y().std()
        ])
