from collections import namedtuple
import random

from gym.wrappers import RescaleAction, TimeLimit
import json
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import Bounds, minimize
import seaborn as sns
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from environments import ARESEAOptimization, ARESEASequential, ResetActuators, ResetActuatorsToDFD


def pack_dataframe(fn):
    def wrapper(env, problem=None):
        observations, incoming, misalignments = fn(env, problem=problem)
        observations = np.array(observations)

        df = pd.DataFrame(np.arange(len(observations)), columns=["step"])
        df["q1"] = observations[:,0]
        df["q2"] = observations[:,1]
        df["q3"] = observations[:,2]
        df["cv"] = observations[:,3]
        df["ch"] = observations[:,4]
        df["mup_x"] = observations[:,5]
        df["mup_y"] = observations[:,6]
        df["sigmap_x"] = observations[:,7]
        df["sigmap_y"] = observations[:,8]
        df["mu_x"] = observations[:,9]
        df["mu_y"] = observations[:,10]
        df["sigma_x"] = observations[:,11]
        df["sigma_y"] = observations[:,12]
        for k in incoming.keys():
            df["incoming_"+k] = float(incoming[k])
        df["misalignment_q1_x"] = misalignments[0]
        df["misalignment_q1_y"] = misalignments[1]
        df["misalignment_q2_x"] = misalignments[2]
        df["misalignment_q2_y"] = misalignments[3]
        df["misalignment_q3_x"] = misalignments[4]
        df["misalignment_q3_y"] = misalignments[5]
        df["misalignment_screen_x"] = misalignments[6]
        df["misalignment_screen_y"] = misalignments[7]

        return df
    
    return wrapper


@pack_dataframe
def run(env, problem=None):
    if problem is not None:
        if "initial" in problem:
            env.unwrapped.next_initial = problem["initial"]
        if "incoming" in problem:
            env.unwrapped.backend.next_incoming = problem["incoming"]
        if "misalignments" in problem:
            env.unwrapped.backend.next_misalignments = problem["misalignments"]
        if "desired" in problem:
            env.unwrapped.next_desired = problem["desired"]

    observations = []

    observation = env.reset()

    observations.append(observation)
    incoming = env.backend._incoming.parameters
    misalignments = env.backend.misalignments

    def optfn(actuators):
        observation, objective, _, _ = env.step(actuators)
        observations.append(observation)
        return objective

    bounds = Bounds(env.action_space.low, env.action_space.high)
    # TODO: Should probably be options={"fatol": 4.5e-11}
    #       Because max pixel_size = 3.3198e-6
    #       Double that (two pixels) -> 6.6396e-06
    #       Squared error would then be 4.408428816e-11
    #       Round to 4.5e-11
    minimize(optfn, observation[:5], method="Nelder-Mead", bounds=bounds, options={"fatol": 4.5e-11, "xatol": 1})

    return observations, incoming, misalignments


def cache_to_file(fn):
    def wrapper(method, **kwargs):
        filename = f".cache_3/{method}.pkl"
        
        try:
            evaluation = pd.read_pickle(filename)
            print(f"Read {method} from cache file")
        except FileNotFoundError:
            evaluation = fn(method, **kwargs)
            evaluation.to_pickle(filename)
        
        return evaluation

    return wrapper


@cache_to_file
def evaluate(method, description=None):
    env = ARESEAOptimization()

    with open("problems_3.json", "r") as f:
        problems = json.load(f)
    
    if "fdf" in method:
        env = ResetActuatorsToDFD(env)
    if "normalize" in method:
        env = RescaleAction(env, -1, 1)

    evaluation = []
    for i, problem in enumerate(tqdm(problems)):
        result = run(env, problem=problem)
        result["problem"] = i
        evaluation.append(result)
    evaluation = pd.concat(evaluation)
    evaluation["method"] = method
    evaluation["model"] = method
    if description is not None:
        evaluation["description"] = description
    
    return evaluation


def main():
    # evaluate("nelder-mead", description="Nelder-Mead Optimiser Starting at 0")
    evaluate("nelder-mead-fdf", description="Nelder-Mead Optimiser Starting at FDF")


if __name__ == "__main__":
    main()
