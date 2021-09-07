from . import simulation

class ARESEAAbsolute(simulation.ARESEACheetah):
    """Variant of the ARES EA environment that uses absolute rather than relative actions."""
    
    def __init__(self):
        super().__init__()
        self.accelerator_action_space = self.accelerator_observation_space["observation"]
    
    def step(self, action):
        action = self.action2accelerator(action)

        self.actuators = action

        info = {"previous_objective": self.history[-1]["objective"]}

        reward = self.compute_reward(
            self.observation["achieved_goal"],
            self.observation["desired_goal"],
            info
        )
        objective = self.compute_objective(
            self.observation["achieved_goal"],
            self.observation["desired_goal"]
        )

        self.finished_steps += 1
        self.history.append({
            "objective": objective,
            "reward": reward,
            "observation": self.observation,
            "action": action
        })

        # done = all(abs(achieved - desired) < 5e-6 for achieved, desired in zip(self.observation["achieved_goal"], self.observation["desired_goal"]))
        done = (abs(self.observation["achieved_goal"] - self.observation["desired_goal"]) < self.target_delta).all()

        # return self.observation2agent(self.observation), self.reward2agent(reward), done, info
        return self.observation2agent(self.observation), self.reward2agent(objective), done, info
