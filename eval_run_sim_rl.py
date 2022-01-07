from collections import namedtuple

from gym.wrappers import RescaleAction, TimeLimit
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from environments import ARESEASequential, ResetActuators, ResetActuatorsToDFD


def load_sequential(model_name, algorithm=TD3, init="dfd"):

    ModelSetup = namedtuple("ModelSetup", ["name","env","model"])

    log_dir = f"models/{model_name}"

    def make_env():
        env = ARESEASequential(
            backend="simulation",
            backendargs={"measure_beam": "direct"}
        )
        if init == "dfd":
            env = ResetActuatorsToDFD(env, k1=10)
        elif init == "zero":
            env = ResetActuators(env)
        elif init == "random":
            pass
        env = TimeLimit(env, max_episode_steps=300)
        env = RescaleAction(env, -1, 1)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(f"{log_dir}/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False

    model = algorithm.load(f"{log_dir}/model")

    return ModelSetup(model_name, env, model)


def pack_dataframe(fn):
    def wrapper(setup, problem=None):
        observations, rewards, incoming, misalignments = fn(setup, problem=problem)
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
        df["reward"] = [np.nan] + rewards
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
def run(setup, problem=None):
    env, model = setup.env, setup.model

    if problem is not None:
        if "initial" in problem:
            env.get_attr("unwrapped")[0].next_initial = problem["initial"]
        if "incoming" in problem:
            env.get_attr("backend")[0].next_incoming = problem["incoming"]
        if "misalignments" in problem:
            env.get_attr("backend")[0].next_misalignments = problem["misalignments"]
        if "desired" in problem:
            env.get_attr("unwrapped")[0].next_desired = problem["desired"]

    observations = []
    rewards = []

    observation = env.reset()

    observations.append(env.unnormalize_obs(observation).squeeze())
    incoming = env.get_attr("backend")[0]._incoming.parameters
    misalignments = env.get_attr("backend")[0].misalignments

    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)

        observations.append(env.unnormalize_obs(observation).squeeze())
        rewards.append(reward[0])

    observations[-1] = env.unnormalize_obs(info[0]["terminal_observation"].squeeze())

    return observations, rewards, incoming, misalignments


def cache_to_file(fn):
    def wrapper(model_name, **kwargs):
        filename = f".cache_3/{model_name}.pkl"
        
        try:
            evaluation = pd.read_pickle(filename)
            print(f"Read {model_name} from cache file")
        except FileNotFoundError:
            evaluation = fn(model_name, **kwargs)
            evaluation.to_pickle(filename)
        except ValueError:
            return None
        
        return evaluation

    return wrapper


@cache_to_file
def evaluate(model_name, algorithm=TD3, method=None, description=None, init="dfd"):
    setup = load_sequential(model_name, algorithm, init)

    with open("problems_3.json", "r") as f:
        problems = json.load(f)

    evaluation = []
    for i, problem in enumerate(tqdm(problems)):
        result = run(setup, problem=problem)
        result["problem"] = i
        evaluation.append(result)
    evaluation = pd.concat(evaluation)
    evaluation["model"] = setup.name
    if method is not None:
        evaluation["method"] = method
    if description is not None:
        evaluation["description"] = description
    
    return evaluation


def main():
    todos = [# {
    #         "method": "initial-random",
    #         "description": "Trained with Random Initial and No Misalignments for 600k Steps",
    #         "models": ["bright-rain-963", "lyric-wave-964", "pleasant-wood-965"],
    #         "algorithm": TD3,
    #         "init": "random"
    #     }, {
    #         "method": "initial-reset",
    #         "description": "Trained 600k With Initial Actuators Set to Zero",
    #         "models": ["faithful-meadow-975", "amber-mountain-976", "ruby-water-977"],
    #         "algorithm": TD3,
    #         "init": "zero"
    #     }, {
    #         "method": "misalignments",
    #         "description": "Quadrupole and Screen Misalignments",
    #         "models": ["ethereal-firefly-972", "royal-planet-973", "clear-armadillo-974"],
    #         "algorithm": TD3,
    #         "init": "zero"
    #     }, {
    #         "method": "6-million",
    #         "description": "Trained for 6M Steps",
    #         "models": ["visionary-blaze-969", "vibrant-leaf-970", "electric-sun-971"],
    #         "algorithm": TD3,
    #         "init": "zero"
    #     }, {
    #         "method": "adjustinit",
    #         "description": "Adjusted Initial Parameters",
    #         "models": ["blooming-vortex-978", "good-serenity-979", "summer-dawn-980"],
    #         "algorithm": TD3,
    #         "init": "zero"
    #     }, {
    #         "method": "msedifferential",
    #         "description": "MSE Objective on Differential Reward",
    #         "models": ["clear-flower-981", "treasured-river-982", "good-disco-983"],
    #         "algorithm": TD3,
    #         "init": "zero"
    #     }
        # }, {
          {
            "method": "msepunish",
            "description": "MSE Objective on Punish Reward",
            "models": ["daily-surf-984", "scarlet-galaxy-985", "polar-disco-986"],
            "algorithm": TD3,
            "init": "zero"
        }, {
            "method": "ourpunish",
            "description": "Our Objective on Punish Reward",
            "models": ["golden-durian-987", "quiet-meadow-988", "fearless-wildflower-989"],
            "algorithm": TD3,
            "init": "zero"
        }, {
            "method": "adjustinit6m",
            "description": "Adjusted Initial Parameters (6 Million Steps)",
            "models": ["sage-plasma-990", "winter-forest-991", "grateful-surf-992"],
            "algorithm": TD3,
            "init": "zero"
        }, {
            "method": "firstppo",
            "description": "First Attempt at PPO (6 Million)",
            "models": ["vague-dawn-993", "dainty-leaf-994", "serene-morning-995"],
            "algorithm": PPO,
            "init": "zero"
        }, {
            "method": "resettodfd",
            "description": "Reset to DFD (with Adjusted Initial)",
            "models": ["polished-donkey-996", "polar-lake-997", "still-deluge-998"],
            "algorithm": TD3,
            "init": "dfd"
        }, {
            "method": "ppo100m",
            "description": "PPO for 100 Million Steps",
            "models": ["curious-paper-999", "ancient-music-1000", "swept-gorge-1001"],
            "algorithm": PPO,
            "init": "zero"
        }]
    
    for todo in todos:
        for model in todo["models"]:
            evaluate(
                model,
                algorithm=todo["algorithm"],
                method=todo["method"],
                description=todo["description"],
                init=todo["init"]
            )


if __name__ == "__main__":
    main()
