import pickle

from gym.wrappers import FlattenObservation, RecordVideo, RescaleAction, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from environment import ARESEA
from utils import CheckpointCallback, FilterAction


misalignments = ARESEA.observation_space["misalignments"].sample()
incoming_parameters = ARESEA.observation_space["incoming"].sample()


def make_env():
    env = ARESEA(misalignments, incoming_parameters)
    env = FilterAction(env, [2,4], replace=0)
    env = TimeLimit(env, max_episode_steps=50)
    env = FlattenObservation(env)
    env = RescaleAction(env, -1, 1)
    env = Monitor(env, info_keywords=("time_reward",))
    return env


def make_eval_env():
    env = ARESEA(misalignments, incoming_parameters)
    env = FilterAction(env, [2,4], replace=0)
    env = TimeLimit(env, max_episode_steps=50)
    env = RecordVideo(env, video_folder="recordings")
    env = FlattenObservation(env)
    env = RescaleAction(env, -1, 1)
    env = Monitor(env, info_keywords=("time_reward",))
    return env


def main():
    wandb.init(project="ares-ea-v2", entity="msk-ipc", sync_tensorboard=True, monitor_gym=True)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, training=False)

    model = PPO("MlpPolicy", env, tensorboard_log=f"log/{wandb.run.name}")

    model.learn(
        total_timesteps=500000,
        eval_env=eval_env,
        eval_freq=4000,
        callback=WandbCallback()
    )

    model.save("model")
    env.save("model_env.pkl")
    with open("model_misalignments.pkl", "wb") as f:
        pickle.dump(misalignments, f)
    with open("model_incoming_parameters.pkl", "wb") as f:
        pickle.dump(incoming_parameters, f)


if __name__ == "__main__":
    main()
