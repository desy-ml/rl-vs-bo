from collections import namedtuple
from datetime import datetime
import logging
from pathlib import Path

from gym.wrappers import RescaleAction, TimeLimit
import json
import numpy as np
import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from environments import ARESEASequential, ResetActuators, ResetActuatorsToDFD
import toolkit


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.WARNING)
formatter = logging.Formatter("[%(asctime)s] - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

Path("log").mkdir(parents=True, exist_ok=True)
logfile = logging.FileHandler("log/evaluate_machine.log")
logfile.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logfile.setFormatter(formatter)
logger.addHandler(logfile)

mail = toolkit.MailHandler(
    ["jan.kaiser@desy.de"],
    name="MSK-IPC Autonomus Accelerator",
    send_history=False
)
mail.setLevel(logging.ERROR)
logger.addHandler(mail)


def load_sequential(model_name, max_episode_steps=30, measure_beam="us", init="dfd"):

    logger.debug(f"Loading setup ({model_name}, max_episode_steps={max_episode_steps}, measure_beam={measure_beam}, init={init})")

    ModelSetup = namedtuple("ModelSetup", ["name","env","model","max_episode_steps","measure_beam"])

    log_dir = f"models/{model_name}"

    def make_env():
        env = ARESEASequential(
            backend="machine",
            backendargs={"measure_beam": measure_beam}
        )
        if init == "dfd":
            env = ResetActuatorsToDFD(env, k1=10)
        elif init == "zero":
            env = ResetActuators(env)
        elif init == "random":
            pass
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = RescaleAction(env, -1, 1)

        env.unwrapped.backend.logger.addHandler(console)
        env.unwrapped.backend.logger.addHandler(logfile)
        env.unwrapped.backend.logger.addHandler(mail)

        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(f"{log_dir}/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False

    model = TD3.load(f"{log_dir}/model")

    return ModelSetup(model_name, env, model, max_episode_steps, measure_beam)


def pack_dataframe(fn):
    def wrapper(setup, problem):
        observations, rewards, beam_images = fn(setup, problem)
        observations = np.array(observations)

        df = pd.DataFrame(np.arange(len(observations)), columns=["step"])
        df["q1"] = observations[:,0]
        df["q2"] = observations[:,1]
        df["cv"] = observations[:,2]
        df["q3"] = observations[:,3]
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
        df["beam_image"] = beam_images

        df["model"] = setup.name
        df["max_episode_steps"] = setup.max_episode_steps
        df["measure_beam"] = setup.measure_beam

        return df
    
    return wrapper


@pack_dataframe
def run(setup, problem):
    env, model = setup.env, setup.model

    if "initial" in problem:
        env.get_attr("unwrapped")[0].next_initial = problem["initial"]
    if "desired" in problem:
        env.get_attr("unwrapped")[0].next_desired = problem["desired"]

    observations = []
    rewards = []
    beam_images = []

    observation = env.reset()
    observations.append(env.unnormalize_obs(observation).squeeze())
    beam_images.append(env.get_attr("backend")[0].last_beam_image)

    env.get_attr("unwrapped")[0].next_initial = "stay"

    with tqdm(total=setup.max_episode_steps, desc="Run") as pbar:
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)

            observations.append(env.unnormalize_obs(observation).squeeze())
            rewards.append(reward.squeeze())
            beam_images.append(info[0]["beam_image"])

            pbar.update(1)

    observations[-1] = env.unnormalize_obs(info[0]["terminal_observation"].squeeze())

    return observations, rewards, beam_images


def evaluate(model_name, directory, method=None, description=None, init="dfd", n=None):
    setup = load_sequential(model_name, init=init)

    if isinstance(n, int):
        n = (0, n)

    with open("problems_3.json", "r") as f:
        problems = json.load(f) if n is None else json.load(f)
        
    Path(directory).mkdir(parents=True, exist_ok=True)

    for i, problem in enumerate(tqdm(problems[n[0]:n[1]], desc="Evaluate"), start=n[0]):
        logger.info(f"Agent {model_name} running problem {i}: Desired = {problem['desired']}")
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result = run(setup, problem=problem)
        result["problem"] = i
        result["model"] = setup.name
        if method is not None:
            result["method"] = method
        if description is not None:
            result["description"] = description
        result.to_pickle(f"{directory}/{model_name}_{i:03d}_{timestamp}.pkl")

        logger.error(f"Agent {model_name} finished running problem {i}")


def main():
    n = (22, 300)
    directory = "machine_studies/evaluation_dummytests"
    todo = {
        "method": "resettodfd",
        "description": "Reset to DFD (with Adjusted Initial)",
        "models": ["polished-donkey-996"], # , "polar-lake-997", "still-deluge-998"],
        "init": "dfd"
    }

    logger.error("Starting evaluation on ARES")

    try:
        for model in todo["models"]:
            evaluate(
                model,
                directory,
                method=todo["method"],
                description=todo["description"],
                init=todo["init"],
                n=n
            )
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {str(e)} -> machine set to safe state")
        from environments.machine import ExperimentalArea
        backend = ExperimentalArea()
        backend._go_to_safe_state()
        raise e

    logger.error("Evaluation has finished.")


if __name__ == "__main__":
    main()
