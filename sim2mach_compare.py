from datetime import datetime

from gym.wrappers import FlattenObservation, Monitor, TimeLimit
import matplotlib.pyplot as plt
import numpy as np

from environments.machine import ARESEAMachine
from environments.simulation import ARESEACheetah


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

env1 = ARESEAMachine()
env1 = TimeLimit(env1, max_episode_steps=50)
env1 = FlattenObservation(env1)
env1 = Monitor(env1, f"experiments/comp_{timestamp}/recording", video_callable=lambda i: True)

env2 = ARESEACheetah()
env2 = TimeLimit(env2, max_episode_steps=50)
env2 = FlattenObservation(env2)
env2 = Monitor(env2, f"experiments/comp_{timestamp}/recording", video_callable=lambda i: True)

env1.reset()

env2.reset()
env2.unwrapped.actuators = env1.unwrapped.actuators

print(f"Intensity = {env1.unwrapped.observation['observation'][0]}")

plt.figure(figsize=(14,5))
plt.subplot(121)
plt.imshow(env1.screen_data, interpolation="None")
plt.subplot(122)
plt.imshow(env2.screen_data, interpolation="None")
plt.show()

env1.close()
env2.close()
