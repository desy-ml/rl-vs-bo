import pickle
import random

from gym.wrappers import FlattenObservation, TimeLimit
import numpy as np
from stable_baselines3 import TD3

from environments.machine import ARESEAMachine


class Agent:

    def __init__(self):
        self.env = ARESEAMachine()
        self.env = TimeLimit(self.env, max_episode_steps=50)
        self.env = FlattenObservation(self.env)

        self.model = TD3.load("models/pretty-jazz-258")

    def move_beam_to(self, mu_x, mu_y, sigma_x, sigma_y):
        user_request = np.array([mu_x, mu_y, sigma_x, sigma_y])

        observation = self.env.reset(goal=user_request)
        done = False
        while not done:
            action, _ = self.model.predict(observation)
            observation, _, done, _ = self.env.step(action)
