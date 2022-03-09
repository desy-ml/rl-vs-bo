from gym.wrappers import RescaleAction, TimeLimit
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb

from environments import ARESEASequential, ResetActuators


def main():
    hyperparameter_defaults = {
        "noise_type": "normal",
        "noise_std": 0.1,
        "learning_rate": 1e-3,
        "buffer_size": 600000,
        "learning_starts": 2000,
        "batch_size": 100,
        "tau": 0.005,
        "gamma": 0.55,
        "gradient_steps": -1,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "net_arch": [64,32]
    }

    wandb.init(
        project="ares-ea-rl-a-long-time-ago",
        entity="msk-ipc",
        config=hyperparameter_defaults,
        sync_tensorboard=True,
        settings=wandb.Settings(start_method="fork")
    )

    def make_env():
        env = ARESEASequential(
            backend="simulation",
            backendargs={"measure_beam": "direct"}
        )
        env = ResetActuators(env)
        env = TimeLimit(env, max_episode_steps=50)
        env = RescaleAction(env, -1, 1)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=wandb.config["gamma"])

    n_actions = env.action_space.shape[-1]
    if wandb.config["noise_type"] == "none":
        noise = None
    elif wandb.config["noise_type"] == "normal":
        noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=np.full(n_actions, wandb.config["noise_std"])
        )

    model = TD3(
        "MlpPolicy",
        env,
        action_noise=noise,
        learning_rate=wandb.config["learning_rate"],
        buffer_size=wandb.config["buffer_size"],
        learning_starts=wandb.config["learning_starts"],
        batch_size=wandb.config["batch_size"],
        tau=wandb.config["tau"],
        gamma=wandb.config["gamma"],
        gradient_steps=wandb.config["gradient_steps"],
        policy_delay=wandb.config["policy_delay"],
        target_policy_noise=wandb.config["target_policy_noise"],
        target_noise_clip=wandb.config["target_noise_clip"],
        policy_kwargs={"net_arch": wandb.config["net_arch"]},
        tensorboard_log=f"log/{wandb.run.name}",
        verbose=1
    )

    done_steps = 0
    STEP_CHUNK = 3000
    while True:
        model.learn(
            total_timesteps=STEP_CHUNK,
            reset_num_timesteps=False,
            log_interval=1,
            tb_log_name="TD3"
        )
        done_steps += STEP_CHUNK
        model.save(f"log/{wandb.run.name}/{done_steps}_model")
        model.save_replay_buffer(f"log/{wandb.run.name}/{done_steps}_replay_buffer")
        env.save(f"log/{wandb.run.name}/{done_steps}_vec_normalize.pkl")


if __name__ == "__main__":
    main()

