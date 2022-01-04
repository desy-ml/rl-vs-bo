from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import pandas as pd
from skopt import dummy_minimize, gp_minimize

from environments import ARESEAOptimization, ResetActuatorsToDFD


def pack_dataframe(fn):
    def wrapper(env, problem=None):
        observations, incoming, misalignments, res = fn(env, problem=problem)
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
        df["misalignment_q1"] = misalignments[0]
        df["misalignment_q2"] = misalignments[1]
        df["misalignment_q3"] = misalignments[2]
        df["misalignment_screen"] = misalignments[4]
        df.loc[:,"res"] = [res] * len(df)

        return df
    
    return wrapper


@pack_dataframe
def run(env, problem=None):
    if problem is not None:
        if "initial" in problem:
            env.next_initial = problem["initial"]
        if "incoming" in problem:
            env.backend.next_incoming = problem["incoming"]
        if "misalignments" in problem:
            env.backend.next_misalignments = problem["misalignments"]
        if "desired" in problem:
            env.next_desired = problem["desired"]

    observations = []

    observation = env.reset()

    observations.append(observation)
    incoming = env.backend._incoming.parameters
    misalignments = env.backend.misalignments

    def optfn(x):
        observation, reward, _, _ = env.step(x)
        observations.append(observation)
        return reward

    bounds = [
        (env.action_space.low[0], env.action_space.high[0]),
        (env.action_space.low[1], env.action_space.high[1]),
        (env.action_space.low[2], env.action_space.high[2]),
        (env.action_space.low[3], env.action_space.high[3]),
        (env.action_space.low[4], env.action_space.high[4])
    ]

    res = gp_minimize(optfn, bounds, x0=list(observation[:5]), n_jobs=-1)

    observation, _, _, _ = env.step(res.x)
    observations.append(observation)

    return observations, incoming, misalignments, res


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
    with open("problems_3.json", "r") as f:
        problems = json.load(f)

    def execute_problem(x):
        i, problem = x

        env = ARESEAOptimization(objective=method[-3:], backendargs={"measure_beam": "direct"})
        env = ResetActuatorsToDFD(env)

        result = run(env, problem=problem)
        result["problem"] = i

        print(f"Executed problem {i}")

    with ProcessPoolExecutor() as executor:
        evaluation = executor.map(execute_problem, enumerate(problems))
        futures.wait(evaluation)

    evaluation = pd.concat(evaluation)
    evaluation["method"] = method
    evaluation["model"] = method
    if description is not None:
        evaluation["description"] = description
    
    return evaluation


def main():
    _ = pd.concat([
        evaluate("bayesian2-mae", description="Bayesian Optimisation with MAE (scipy-optimize)"),
        evaluate("bayesian2-mse", description="Bayesian Optimisation with MSE (scipy-optimize)"),
        evaluate("bayesian2-log", description="Bayesian Optimisation with Our Log Objective (scipy-optimize)")
    ])


if __name__ == "__main__":
    main()
