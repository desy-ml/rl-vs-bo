from gym.wrappers import FlattenObservation
import torch
from torch import distributions
from torch import nn
from torch import optim

from environments.absolute import ARESEAAbsolute


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


class OneShotPolicyGradient:

    def __init__(self, env, batch_size=64):
        self.env = env
        self.batch_size = batch_size

        self.actor = GaussianActor(env.observation_space.shape, env.action_space.shape)
        self.optimizer = optim.Adam(self.actor.parameters)
    
    def learn(self, n_steps=1000):
        step = 0
        while step < n_steps:
            observations = torch.zeros(self.batch_size, self.env.observation_space.shape)
            actions = torch.zeros(self.batch_size, self.env.action_space.shape)
            rewards = torch.zeros(self.batch_size)
            for i in range(self.batch_size):
                observation = env.reset()
                observation = torch.tensor(observation, dtype=torch.float32)
                observations[i] = observation

                observation = torch.unsqueeze(observation, 0)
                action = self.actor(observation)
                action = torch.squeeze(action)
                actions[i] = action
                
                action = action.numpy()
                _, reward, _, _ = env.step(action)
                rewards[i] = reward
            
            

            step += self.batch_size


if __name__ == "__main__":
    env = ARESEAAbsolute()
    env = FlattenObservation(env)
