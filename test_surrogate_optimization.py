import pickle
import time

import accelerator_environments
import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


# Setup environments
surrogate = gym.make("ARESEA-JOSS-v0")
machine = gym.make("ARESEA-Machine-v0")

surrogate_observation = surrogate.reset()
machine_observation = machine.reset()

# Record snapshot before optimisation
before = {"surrogate": {"screen_data": surrogate.screen_data,
                        "screen_rendered": surrogate.render(),
                        "observation": surrogate_observation},
          "machine": {"screen_data": machine.screen_data,
                      "screen_rendered": machine.render(),
                      "observation": machine_observation}}

# Optimise surrogate and then set found values on machine
t1 = time.time()
bounds = optimize.Bounds(surrogate.optimization_space.low, surrogate.optimization_space.high)
x = optimize.minimize(fun=surrogate.objective_function,
                      x0=surrogate.initial_actuator_values(),
                      bounds=bounds)
t2 = time.time()

machine.objective_function(x.x)
t3 = time.time()

# Record snapshot after optimisation
after = {"surrogate": {"screen_data": surrogate.screen_data,
                       "screen_rendered": surrogate.render(),
                       "observation": np.concatenate([surrogate.beam_parameters(), x.x])},
         "machine": {"screen_data": machine.screen_data,
                     "screen_rendered": machine.render(),
                     "observation": np.concatenate([machine.beam_parameters(), machine.read_actuators()])}}

# Save results of this test
record = {"before": before, "after": after, "x": x, "t1": t1, "t2": t2, "t3":t3}
path = "test_surrogate_optimization_" + time.strftime("%Y%m%d%H%M%S") + ".pkl"
with open(path, "wb") as file:
    pickle.dump(record, file)

# Just some plotting so the operator can immediately get some idea if the script succeeded
screen_extent = (-surrogate.screen_resolution[0] * surrogate.pixel_size[0] / 2,
                 surrogate.screen_resolution[0] * surrogate.pixel_size[0] / 2,
                 -surrogate.screen_resolution[1] * surrogate.pixel_size[1] / 2,
                 surrogate.screen_resolution[1] * surrogate.pixel_size[1] / 2)

plt.figure(figsize=(10,6))
plt.subplot(221)
plt.title("Before (Surrogate)")
plt.imshow(before["surrogate"]["screen_rendered"], interpolation="None", extent=screen_extent)
plt.subplot(222)
plt.title("After (Surrogate)")
plt.imshow(after["surrogate"]["screen_rendered"], interpolation="None", extent=screen_extent)
plt.subplot(223)
plt.title("Before (Machine)")
plt.imshow(before["machine"]["screen_rendered"], interpolation="None", extent=screen_extent)
plt.subplot(224)
plt.title("After (Machine)")
plt.imshow(after["machine"]["screen_rendered"], interpolation="None", extent=screen_extent)
plt.tight_layout()
plt.show()
