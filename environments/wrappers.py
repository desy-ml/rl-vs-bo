from datetime import datetime
import os

import gym
from gym import spaces
import numpy as np
from PIL import Image
from numpy.lib.arraysetops import isin


class ScaleAction(gym.ActionWrapper):
    """Rescale a continuous action by a factor of `scale`."""
    
    def __init__(self, env, scale):
        super().__init__(env)
        
        self.scale = scale
        self.action_space = spaces.Box(low=env.action_space.low / scale,
                                       high=env.action_space.high / scale)
        
    def action(self, action):
        return action * self.scale
    
    def render(self, **kwargs):
        return self.env.render(action_scale=self.scale, **kwargs)


class NormalizeAction(ScaleAction):
    """Normalise a continuous action."""

    def __init__(self, env):
        super().__init__(env, env.action_space.high)


class ScaleObservation(gym.ObservationWrapper):
    """Rescale a continuous observation by a factor of `scale`."""

    def __init__(self, env, scale):
        super().__init__(env)

        self.scale = scale

        if isinstance(env.unwrapped, gym.GoalEnv):
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(
                    low=env.observation_space["observation"].low / scale["observation"],
                    high=env.observation_space["observation"].high / scale["observation"]
                ),
                "desired_goal": spaces.Box(
                    low=env.observation_space["desired_goal"].low / scale["desired_goal"],
                    high=env.observation_space["desired_goal"].high / scale["desired_goal"]
                ),
                "achieved_goal": spaces.Box(
                    low=env.observation_space["achieved_goal"].low / scale["achieved_goal"],
                    high=env.observation_space["achieved_goal"].high / scale["achieved_goal"]
                )
            })
        else:
            self.observation_space = spaces.Box(
                low=env.observation_space.low / scale,
                high=env.observation_space.high / scale
            )
    
    def observation(self, observation):
        if isinstance(self.env.unwrapped, gym.GoalEnv):
            return {
                "observation": observation["observation"] / self.scale["observation"],
                "desired_goal": observation["desired_goal"] / self.scale["desired_goal"],
                "achieved_goal": observation["achieved_goal"] / self.scale["achieved_goal"]
            }
        else:
            return observation / self.scale
    
    def render(self, **kwargs):
        return self.env.render(observation_scale=self.scale, **kwargs)


class NormalizeObservation(ScaleObservation):
    """Normalise a continuous observation."""

    def __init__(self, env):
        if isinstance(env.unwrapped, gym.GoalEnv):
            scale = {k: v.high for k, v in env.observation_space.spaces.items()}
            super().__init__(env, scale=scale)
        else:
            super().__init__(env, env.observation_space.high)


class ScaleReward(gym.RewardWrapper):
    """Rescale reward by a factor of `scale`."""

    def __init__(self, env, scale):
        super().__init__(env)

        self.scale = scale
    
    def reward(self, reward):
        return reward / self.scale
    
    def render(self, **kwargs):
        return self.env.render(reward_scale=self.scale, **kwargs)


class ScaleActuators(gym.Wrapper):
    """Rescale continuous actuators by a factor of `scale`."""

    def __init__(self, env, scale):
        super().__init__(env)

        self.scale = scale
        self.optimization_space = spaces.Box(low=env.optimization_space.low / scale,
                                             high=env.optimization_space.high / scale)
    
    def objective_function(self, actuators):
        return self.env.objective_function(actuators * self.scale)
    
    def render(self, **kwargs):
        n_non_action_observables = self.observation_space.shape[0] - self.action_space.shape[0]
        if isinstance(self.scale, np.ndarray):
            render_scale = np.concatenate([np.ones(n_non_action_observables), self.scale])
        else:
            render_scale = np.array([1]*n_non_action_observables, [self.scale]*5)

        return self.env.render(observation_scale=render_scale, **kwargs)


class SaveFinalRender(gym.Wrapper):
    """Wrapper to save the final render of an environment."""

    def __init__(self, env, directory):
        super().__init__(env)

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self.directory = directory
        self.env_id = "(unknown)" if self.env.spec is None else self.env.spec.id

        self.done = False
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.done = done
        return observation, reward, done, info

    def reset(self, **kwargs):
        if self.done:
            self.write_render()
            self.done = False
        return self.env.reset(**kwargs)
    
    def close(self):
        if self.done:
            self.write_render()
        return self.env.close()
    
    def write_render(self):
        nowstring = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"FinalRender-{self.env_id}-{nowstring}.png"
        path = os.path.join(self.directory, filename)

        render = self.env.render(mode="rgb_array")
        Image.fromarray(render).save(path)
