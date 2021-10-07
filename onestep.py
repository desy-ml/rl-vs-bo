from datetime import datetime
import pickle

from gym import spaces
import numpy as np
import torch
from torch import distributions
from torch import nn
from torch import optim

from environments import machine, simulation, utils


class PseudoEnv:

    n_observations = 13
    n_actuators = 5

    actuator_space = spaces.Box(
        low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32),
        high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32)
    )
    goal_space = spaces.Box(
        low=np.array([-2e-3, -2e-3, 0, 0], dtype=np.float32),
        high=np.array([2e-3, 2e-3, 5e-4, 5e-4], dtype=np.float32)
    )

    def __init__(self, backend="simulation", random_incoming=False, random_initial=False, beam_parameter_method="us"):
        self.backend = backend
        self.random_incoming = random_incoming
        self.random_initial = random_initial
        self.beam_parameter_method = beam_parameter_method

        if self.backend == "simulation":
            self.accelerator = simulation.ExperimentalArea()
        elif self.backend == "machine":
            self.accelerator = machine.ExperimentalArea()
        else:
            raise ValueError(f"There is no \"{backend}\" backend!")
    
    def reset(self, desired=None):
        if self.random_incoming:
            self.accelerator.randomize_incoming()
        
        if self.random_initial:
            self.accelerator.actuators = self.actuator_space.sample()
        
        self.desired = desired if desired is not None else self.goal_space.sample()

        self._screen_data = self.accelerator.capture_clean_beam()
        self.achieved = self.beam_parameters

        observation = np.concatenate([self.accelerator.actuators, self.desired, self.achieved])

        return observation

    
    def track(self, actuators):
        self.accelerator.actuators = actuators
        self._screen_data = self.accelerator.capture_clean_beam()
        self.achieved = self.beam_parameters
                
        return self.achieved
    
    @property
    def beam_parameters(self):
        if self.beam_parameter_method == "direct":
            return self._read_beam_parameters_from_simulation()
        else:
            return utils.compute_beam_parameters(
                self._screen_data,
                self.accelerator.pixel_size*self.accelerator.binning,
                method=self.beam_parameter_method)
    
    def _read_beam_parameters_from_simulation(self):
        return np.array([
            self.accelerator.segment.AREABSCR1.read_beam.mu_x,
            self.accelerator.segment.AREABSCR1.read_beam.mu_y,
            self.accelerator.segment.AREABSCR1.read_beam.sigma_x,
            self.accelerator.segment.AREABSCR1.read_beam.sigma_y
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
    print("Main started")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    n_batches = 9375
    batch_size = 64

    def make_env():
        return PseudoEnv(
            backend="simulation", 
            random_incoming=True, 
            random_initial=True, 
            beam_parameter_method="direct"
        )

    simulations = [make_env() for _ in range(batch_size)]

    policy = GaussianActor(simulations[0].n_observations, simulations[0].n_actuators)
    optimizer = optim.Adam(policy.parameters())

    history = []
    
    print("Setup done, starting training.")

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
