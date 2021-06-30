from gym.spaces import Box
import numpy as np

from accelerator_environments.envs.ares import aresea_v1


class ARESEA(aresea_v1.ARESEA):
    """
    ARESEA tasks the agent with centering and focusing an electron beam on a diagnostic screen
    downstream from the experimental in ARES (Accelerator Research Experiment at SINBAD).
    
    To achieve beam focusing and centering, the agent may control a quadrupole triplet and two
    correctors, a vertical and a horizontal one. Rewards are given according to the improvement of
    the beam score computed as the sum of beam parameters :math:`(\mu_x, \mu_y, \sigma_x, \sigma_y)`.
    """

    observation_space = Box(low=np.array([-4e-3, -4e-3, 0, 0, 0, -1e-3/50, -1e-3/50, -1e-3,
                                          -1e-3, -10, -10, -10, -1e-3, -1e-3], dtype=np.float32),
                            high=np.array([4e-3, 4e-3, 4e-4, 4e-4, 1e5, 1e-3/50, 1e-3/50, 1e-3,
                                           1e-3, 10, 10, 10, 1e-3, 1e-3], dtype=np.float32))
    action_space = Box(low=observation_space.low[-9:] * 0.1,
                       high=observation_space.high[-9:] * 0.1)
    optimization_space = Box(low=observation_space.low[-9:],
                             high=observation_space.high[-9:])
    
    target = np.array([0, 0, 0, 0])
    goal = np.array([1e-5, 1e-5, 5e-5, 5e-5])
    
    def plot_actions(self, ax, action_scale=1):
        actions = np.stack([record["action"] for record in self.history])
        actions = actions / action_scale

        ax.set_title("Actions")
        for i, name in enumerate(["$C_{h1}$", "$C_{v1}$", "$C_{h2}$", "$C_{v2}$", "$Q_1$", "$Q_2$",
                                  "$Q_3$", "$C_{v3}$", "$C_{h3}$"]):
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
                                  "$C_{h1}$", "$C_{v1}$", "$C_{h2}$", "$C_{v2}$", "$Q_1$", "$Q_2$",
                                  "$Q_3$", "$C_{v3}$", "$C_{h3}$"]):
            ax.plot(observations[:,i], label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (in Agent View)")
        ax.legend(loc="upper right")
        ax.grid()
