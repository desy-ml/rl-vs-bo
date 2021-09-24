from datetime import datetime
import pickle

import cheetah
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import distributions
from torch import nn
from torch import optim

from environments import ARESlatticeStage3v1_9 as lattice
from environments import utils


class Simulation:

    n_observations = 13
    n_actuators = 5

    def __init__(self):
        screen_resolution = (2448, 2040)
        pixel_size = (3.3198e-6, 2.4469e-6)

        cell = utils.subcell_of(lattice.cell, "AREASOLA1", "AREABSCR1")

        self.segment = cheetah.Segment.from_ocelot(cell)
        self.segment.AREABSCR1.resolution = screen_resolution
        self.segment.AREABSCR1.pixel_size = pixel_size
        self.segment.AREABSCR1.is_active = True

        self.segment.AREABSCR1.binning = 4

        self.actuator_space = spaces.Box(
            low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32),
            high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32)
        )
        self.goal_space = spaces.Box(
            low=np.array([-2e-3, -2e-3, 0, 0], dtype=np.float32),
            high=np.array([2e-3, 2e-3, 5e-4, 5e-4], dtype=np.float32)
        )
    
    def reset(self, incoming=None, initial_actuators=None, desired=None):
        if incoming is None:
            self.incoming = cheetah.Beam.make_random(
                n=int(1e5),
                mu_x=np.random.uniform(-3e-3, 3e-3),
                mu_y=np.random.uniform(-3e-4, 3e-4),
                mu_xp=np.random.uniform(-1e-4, 1e-4),
                mu_yp=np.random.uniform(-1e-4, 1e-4),
                sigma_x=np.random.uniform(0, 2e-3),
                sigma_y=np.random.uniform(0, 2e-3),
                sigma_xp=np.random.uniform(0, 1e-4),
                sigma_yp=np.random.uniform(0, 1e-4),
                sigma_s=np.random.uniform(0, 2e-3),
                sigma_p=np.random.uniform(0, 5e-3),
                energy=np.random.uniform(80e6, 160e6)
            )
        else:
            self.incoming = incoming
            
        if initial_actuators is None:
            initial_actuators = self.actuator_space.sample()
        
        if desired is None:
            self.desired = self.goal_space.sample()
        else:
            self.desired = desired

        achieved = self.track(initial_actuators)

        observation = np.concatenate([initial_actuators, self.desired, achieved])

        return observation
    
    def track(self, actuators):
        self.segment.AREAMQZM1.k1, self.segment.AREAMQZM2.k1, self.segment.AREAMQZM3.k1 = actuators[:3]
        self.segment.AREAMCVM1.angle, self.segment.AREAMCHM1.angle = actuators[3:]

        _ = self.segment(self.incoming)
        
        return np.array([
            self.segment.AREABSCR1.read_beam.mu_x,
            self.segment.AREABSCR1.read_beam.mu_y,
            self.segment.AREABSCR1.read_beam.sigma_x,
            self.segment.AREABSCR1.read_beam.sigma_y
        ])


class GaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.log_std = nn.Parameter(0.5 * torch.ones(act_dim))
        self.mu_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, act_dim)
        )
    
    def forward(self, observation, action=None):
        mu = self.mu_net(observation)
        std = torch.exp(self.log_std)
        pi = distributions.Normal(mu, std)

        if action is None:
            return pi
        else:
            log_probs = pi.log_prob(action).sum(axis=-1)
            return pi, log_probs


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    n_batches = 9375
    batch_size = 64

    simulations = [Simulation() for _ in range(batch_size)]

    policy = GaussianActor(simulations[0].n_observations, simulations[0].n_actuators)
    optimizer = optim.Adam(policy.parameters())

    history = []

    for i in range(n_batches):
        # Rollout
        observations = [s.reset() for s in simulations]

        observation_factor = np.concatenate([
            simulations[0].actuator_space.high,
            simulations[0].goal_space.high,
            simulations[0].goal_space.high
        ])
        normalized_observations = [o / observation_factor for o in observations]
        normalized_observations = torch.tensor(normalized_observations, dtype=torch.float32)

        normalized_actuators = policy(normalized_observations).sample()
        actuators = normalized_actuators.detach().numpy() * simulations[0].actuator_space.high

        achieveds = [s.track(a) for s, a in zip(simulations, actuators)]
        desireds = [s.desired for s in simulations]

        def objective_fn(achieved, desired):
            offset = achieved - desired
            weights = np.array([1, 1, 2, 2])

            return np.log((weights * np.abs(offset)).sum())

        objectives = [objective_fn(a, d) for a, d in zip(achieveds, desireds)]
        objectives = torch.tensor(objectives, dtype=torch.float32)

        # Update
        policy_distributions = policy(normalized_observations)
        log_probs = policy_distributions.log_prob(normalized_actuators).sum(axis=-1)

        mean = objectives.mean()
        std = objectives.std().clamp_min(1e-12)
        normalized_objectives = (objectives - mean) / std

        loss = (log_probs * normalized_objectives).mean()

        policy.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {i}: loss ={float(loss)}, objective = {float(objectives.mean())}")
        history.append({"loss": float(loss), "objective": float(objectives.mean())})

    # plt.title("Loss")
    # plt.plot([r["objective"] for r in history])
    # plt.show()

    model_file = f"models/onestep-model-{timestamp}.pkl"
    torch.save(policy, model_file)
    print(f"Saved \"{model_file}\"")

    history_file = f"models/onestep-history-{timestamp}.pkl"
    with open(history_file, "wb") as f:
        pickle.dump(history, f)
    print(f"Saved \"{history_file}\"")
