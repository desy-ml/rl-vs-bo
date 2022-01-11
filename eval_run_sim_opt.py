from gym.wrappers import RescaleAction
import json
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize
from tqdm import tqdm

from environments import ARESEAOptimization, ResetActuatorsToDFD


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
    # minimize(optfn, observation[:5], method="Nelder-Mead", bounds=bounds, options={"fatol": 4.5e-11, "xatol": 1})
    # minimize(optfn, observation[:5], method="Powell", bounds=bounds, options={"ftol": 4.5e-11, "xtol": 1})
    # minimize(optfn, observation[:5], method="COBYLA", bounds=bounds, options={"tol": 4.5e-11, "rhobeg": 1})
    minimize(optfn, observation[:5], method="COBYLA", bounds=bounds, options={"tol": 4.5e-11, "rhobeg": 1e-3})

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
    env = ARESEAOptimization(objective="mse", backendargs={"measure_beam": "direct"})
    if "mae" in method:
        env = ARESEAOptimization(objective="mae", backendargs={"measure_beam": "direct"})
    elif "log" in method:
        env = ARESEAOptimization(objective="log", backendargs={"measure_beam": "direct"})

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
    # evaluate("nelder-mead-fdf", description="Nelder-Mead Optimiser Starting at FDF")
    # evaluate("nelder-mead-fdf-mae", description="Nelder-Mead Optimiser Starting at FDF (MAE)")
    # evaluate("nelder-mead-fdf-log", description="Nelder-Mead Optimiser Starting at FDF (LOG)")
    # evaluate("powell-fdf", description="Powell Optimiser Starting at FDF")
    # evaluate("powell-fdf-mae", description="Powell Optimiser Starting at FDF (MAE)")
    # evaluate("powell-fdf-log", description="Powell Optimiser Starting at FDF (LOG)")
    evaluate("cobyla-fdf-mae", description="COBYLA Optimiser Starting at FDF (MAE)")
    evaluate("cobyla-fdf-mse", description="COBYLA Optimiser Starting at FDF (MSE)")
    evaluate("cobyla-fdf-log", description="COBYLA Optimiser Starting at FDF (LOG)")
    evaluate("cobyla-fdf-mae-1e-3", description="COBYLA Optimiser Starting at FDF and rhobeg=1e-3 (MAE)")
    evaluate("cobyla-fdf-mse-1e-3", description="COBYLA Optimiser Starting at FDF and rhobeg=1e-3 (MSE)")
    evaluate("cobyla-fdf-log-1e-3", description="COBYLA Optimiser Starting at FDF and rhobeg=1e-3 (LOG)")


if __name__ == "__main__":
    main()
