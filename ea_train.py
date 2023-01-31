from functools import partial

import gym
import wandb
from gym.wrappers import (
    FilterObservation,
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from backend import CheetahBackend
from environment import EATransverseTuning
from utils import FilterAction, save_config


def main() -> None:
    config = {
        "action_mode": "delta",
        "batch_size": 100,
        "beam_distance_ord": 2,
        "gamma": 0.99,
        "filter_action": None,
        "filter_observation": None,
        "frame_stack": None,
        "incoming_mode": "random",
        "incoming_values": None,
        "learning_rate": 0.0003,
        "logarithmic_beam_distance": False,
        "magnet_init_mode": "random",
        "magnet_init_values": None,
        "max_misalignment": 5e-4,
        "max_quad_delta": 72 * 0.1,
        "max_steerer_delta": 6.1782e-3 * 0.1,
        "misalignment_mode": "random",
        "misalignment_values": None,
        "n_envs": 40,
        "n_steps": 100,
        "normalize_beam_distance": True,
        "normalize_observation": True,
        "normalize_reward": True,
        "rescale_action": (-1, 1),
        "reward_mode": "feedback",
        "sb3_device": "auto",
        "target_beam_mode": "random",
        "target_beam_values": None,
        "target_mu_x_threshold": 20e-6,
        "target_mu_y_threshold": 20e-6,
        "target_sigma_x_threshold": 20e-6,  # 20e-6 m are close to screen resolution
        "target_sigma_y_threshold": 20e-6,
        "threshold_hold": 3,
        "time_limit": 50,
        "vec_env": "subproc",
        "w_beam": 1.0,
        "w_done": 10.0,
        "w_mu_x": 1.0,
        "w_mu_x_in_threshold": 0.0,
        "w_mu_y": 1.0,
        "w_mu_y_in_threshold": 0.0,
        "w_on_screen": 0.0,
        "w_sigma_x": 1.0,
        "w_sigma_x_in_threshold": 0.0,
        "w_sigma_y": 1.0,
        "w_sigma_y_in_threshold": 0.0,
        "w_time": 0.0,
    }

    train(config)


def train(config: dict) -> None:
    # Setup wandb
    wandb.init(
        project="ares-ea-v2",
        entity="msk-ipc",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
        dir=".wandb",
    )
    config = dict(wandb.config)
    config["run_name"] = wandb.run.name

    # Setup environments
    if config["vec_env"] == "dummy":
        env = DummyVecEnv([partial(make_env, config) for _ in range(config["n_envs"])])
    elif config["vec_env"] == "subproc":
        env = SubprocVecEnv(
            [partial(make_env, config) for _ in range(config["n_envs"])]
        )
    else:
        raise ValueError(f"Invalid value \"{config['vec_env']}\" for dummy")
    eval_env = DummyVecEnv([partial(make_env, config, record_video=True)])

    if config["normalize_observation"] or config["normalize_reward"]:
        env = VecNormalize(
            env,
            norm_obs=config["normalize_observation"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=config["normalize_observation"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
            training=False,
        )

    # Train
    model = PPO(
        "MlpPolicy",
        env,
        device=config["sb3_device"],
        gamma=config["gamma"],
        tensorboard_log=f"log/{config['run_name']}",
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
    )

    eval_callback = EvalCallback(eval_env, eval_freq=1_000, n_eval_episodes=100)
    wandb_callback = WandbCallback()

    model.learn(
        total_timesteps=5_000_000,
        callback=[eval_callback, wandb_callback],
    )

    model.save(f"models/{wandb.run.name}/model")
    if config["normalize_observation"] or config["normalize_reward"]:
        env.save(f"models/{wandb.run.name}/vec_normalize.pkl")
    save_config(config, f"models/{wandb.run.name}/config")


def make_env(config: dict, record_video: bool = False) -> gym.Env:
    cheetah_backend = CheetahBackend(
        incoming_mode=config["incoming_mode"],
        incoming_values=config["incoming_values"],
        max_misalignment=config["max_misalignment"],
        misalignment_mode=config["misalignment_mode"],
        misalignment_values=config["misalignment_values"],
    )

    env = EATransverseTuning(
        backend=cheetah_backend,
        action_mode=config["action_mode"],
        beam_distance_ord=config["beam_distance_ord"],
        logarithmic_beam_distance=config["logarithmic_beam_distance"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=config["magnet_init_values"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        normalize_beam_distance=config["normalize_beam_distance"],
        reward_mode=config["reward_mode"],
        target_beam_mode=config["target_beam_mode"],
        target_beam_values=config["target_beam_values"],
        target_mu_x_threshold=config["target_mu_x_threshold"],
        target_mu_y_threshold=config["target_mu_y_threshold"],
        target_sigma_x_threshold=config["target_sigma_x_threshold"],
        target_sigma_y_threshold=config["target_sigma_y_threshold"],
        threshold_hold=config["threshold_hold"],
        w_beam=config["w_beam"],
        w_mu_x=config["w_mu_x"],
        w_mu_x_in_threshold=config["w_mu_x_in_threshold"],
        w_mu_y=config["w_mu_y"],
        w_mu_y_in_threshold=config["w_mu_y_in_threshold"],
        w_on_screen=config["w_on_screen"],
        w_sigma_x=config["w_sigma_x"],
        w_sigma_x_in_threshold=config["w_sigma_x_in_threshold"],
        w_sigma_y=config["w_sigma_y"],
        w_sigma_y_in_threshold=config["w_sigma_y_in_threshold"],
        w_time=config["w_time"],
    )
    if config["filter_observation"] is not None:
        env = FilterObservation(env, config["filter_observation"])
    if config["filter_action"] is not None:
        env = FilterAction(env, config["filter_action"], replace=0)
    if config["time_limit"] is not None:
        env = TimeLimit(env, config["time_limit"])
    env = FlattenObservation(env)
    if config["frame_stack"] is not None:
        env = FrameStack(env, config["frame_stack"])
    if config["rescale_action"] is not None:
        env = RescaleAction(
            env, config["rescale_action"][0], config["rescale_action"][1]
        )
    env = Monitor(env)
    if record_video:
        env = RecordVideo(env, video_folder=f"recordings/{config['run_name']}")
    return env


if __name__ == "__main__":
    main()
