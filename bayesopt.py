from typing import Optional, Union
import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import (
    UpperConfidenceBound,
    ExpectedImprovement,
    qExpectedImprovement,
)
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means.mean import Mean
import gym
from gym.spaces.utils import unflatten
import numpy as np

config = {
    "action_mode": "direct",
    "gamma": 0.99,
    # "filter_action": [0, 1, 3],
    "filter_action": None,
    "filter_observation": None,
    "frame_stack": None,
    "incoming_mode": "random",
    "incoming_values": np.array(
        [1.18e08, 1e-4, 0, 1e-4, 0, 5e-05, 6e-06, 4e-04, 3e-05, 9e-06, 7e-04],
        dtype="float32",
    ),
    "magnet_init_mode": "constant",
    "magnet_init_values": np.array([10, -10, 0, 10, 0]),
    "misalignment_mode": "constant",
    "misalignment_values": np.zeros(8),
    "n_envs": 40,
    "normalize_observation": True,
    "normalize_reward": True,
    "rescale_action": (-3, 3),
    "reward_mode": "logl1",
    "sb3_device": "auto",
    "target_beam_mode": "constant",
    "target_beam_values": np.zeros(4),
    "target_mu_x_threshold": 1e-5,
    "target_mu_y_threshold": 1e-5,
    "target_sigma_x_threshold": 1e-5,
    "target_sigma_y_threshold": 1e-5,
    "threshold_hold": 5,
    "time_limit": 50000,
    "vec_env": "subproc",
    "w_done": 0.0,
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


def calculate_objective(env, observation, reward, obj="reward", w_on_screen=10):
    """A wrapper for getting objective not (yet) defined in the class

    Could be interesting objectives:
        worstlogl1: take the log of the worst L1 value of the beam parameters
        logmae: as used before log(MAE(current_beam - target_beam))
    """
    if obj == "reward":
        objective = reward
    else:
        obs = unflatten(env.unwrapped.observation_space, observation)
        cb = obs["beam"]
        tb = obs["target"]

        if obj == "clipped_l1":
            logl1 = -np.log(np.abs(cb - tb))
            # resolution limit -log(3e-6) ~ 12.5
            objective = np.clip(logl1, None, 12.5).sum()

        elif obj == "worstl1":
            l1 = -np.abs(cb - tb)
            objective = np.min(l1)
        elif obj == "worstlogl1":
            logl1 = -np.log(np.abs(cb - tb))
            objective = np.min(logl1)
        elif obj == "worstl2":
            l2 = -((cb - tb) ** 2)
            objective = np.min(l2)
        elif obj == "logmae":
            mae = np.mean(np.abs(cb - tb))
            objective = -np.log(mae)
        elif obj == "mae":
            objective = np.mean(np.abs(cb - tb))
        else:
            raise NotImplementedError(f"Objective {obj} not known")
    on_screen_reward = 1 if env.is_beam_on_screen() else -1
    objective += w_on_screen * on_screen_reward
    return objective


def scale_action(env, observation, filter_action=None):
    """Scale the observed magnet settings to proper action values"""
    magnet_values = unflatten(env.unwrapped.observation_space, observation)["magnets"]
    action_values = []
    if filter_action is None:
        filter_action = [0, 1, 2, 3, 4]

    for i, act in enumerate(filter_action):
        scaled_low = env.action_space.low[i]
        scaled_high = env.action_space.high[i]
        low = env.unwrapped.action_space.low[act]
        high = env.unwrapped.action_space.high[act]
        action = scaled_low + (scaled_high - scaled_low) * (
            (magnet_values[act] - low) / (high - low)
        )
        action_values.append(action)
    return action_values


def get_new_bound(env, current_action, stepsize):
    bounds = np.array([env.action_space.low, env.action_space.high])
    bounds = stepsize * bounds + current_action
    bounds = np.clip(bounds, env.action_space.low, env.action_space.high)
    return bounds


def get_next_samples(
    X: torch.Tensor,
    Y: torch.Tensor,
    best_y: Union[float, torch.Tensor],
    bounds: torch.Tensor,
    n_points: int = 1,
    acquisition: str = "EI",
    fixparam: Optional[dict] = None,
    mean_module: Optional[Mean] = None,
):
    """
    Suggest Next Sample for BO
    """
    gp = SingleTaskGP(X, Y, mean_module=mean_module)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    # Exclude fixed hyperparameters
    if fixparam is not None:
        if "lengthscale" in fixparam.keys():
            gp.covar_module.base_kernel.lengthscale = fixparam["lengthscale"]
            gp.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        if "noise_var" in fixparam.keys():
            gp.likelihood.noise_covar.noise = fixparam["noise_var"]
            gp.likelihood.noise_covar.raw_noise.requires_grad = False
        if "mean_constant" in fixparam.keys():
            gp.mean_module.constant = fixparam["mean_constant"]
            gp.mean_module.raw_constant.requires_grad = False
        if "scale" in fixparam.keys():
            gp.covar_module.output_scale = fixparam["scale"]
            gp.covar_module.raw_outputscale.requires_grad = False

    # Fit GP if any parameter is not fixed
    if any(param.requires_grad for _, param in gp.named_parameters()):
        fit_gpytorch_model(mll)

    if acquisition == "EI":
        acq = ExpectedImprovement(model=gp, best_f=best_y)
    elif acquisition == "qEI":
        acq = qExpectedImprovement(model=gp, best_f=best_y)
    elif acquisition == "UCB":
        acq = UpperConfidenceBound(gp, beta=0.1)

    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=n_points,
        num_restarts=10,
        raw_samples=128,
        options={"maxiter": 200},
    )

    return candidates


def bo_optimize(
    env: gym.Env,
    budget=100,
    init_x=None,
    init_samples=5,
    acq="EI",
    obj="reward",
    filter_action=None,
):
    """Complete BO loop, not quite fit into the ea_optimize logic yet"""
    observation = env.reset()
    x_dim = env.action_space.shape[0]
    bounds = torch.tensor(
        np.array([env.action_space.low, env.action_space.high]), dtype=torch.float32
    )
    # Initialization
    if init_x is not None:  # From fix starting points
        X = torch.tensor(init_x.reshape(-1, x_dim), dtype=torch.float32)
    else:  # Random Initialization
        action_i = scale_action(env, observation, filter_action)
        X = torch.tensor([action_i], dtype=torch.float32)
        for i in range(init_samples - 1):
            X = torch.cat([X, torch.tensor([env.action_space.sample()])])
    # Sample initial Y
    Y = torch.empty((X.shape[0], 1))
    for i, action in enumerate(X):
        action = action.detach().numpy()
        observation, reward, done, info = env.step(action)
        objective = calculate_objective(env, observation, reward, obj=obj)
        # _, reward, done, _ = env.step(action)
        Y[i] = torch.tensor(objective)

    # In the loop
    for i in range(budget):
        action = get_next_samples(X, Y, Y.max(), bounds, n_points=1, acquisition=acq)
        observation, _, done, _ = env.step(action.detach().numpy().flatten())
        objective = calculate_objective(env, observation, reward, obj=obj)

        # append data
        X = torch.cat([X, action])
        Y = torch.cat([Y, torch.tensor([[objective]], dtype=torch.float32)])

        if done:
            print(f"Optimization succeeds in {i} steps")
            break

    return X.detach().numpy(), Y.detach().numpy()


# Use a NN as GP prior for BO
class SimpleBeamPredictNN(nn.Module):
    """A simple FCNN to predict the output beam parameters assuming centered incoming beam"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(
                5,
                8,
                bias=True,
            ),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )

    def forward(self, x):
        return self.layers(x)


class BeamNNPrior(Mean):
    """Use a NN as GP prior mean function
    The NN predicts the output beam, which is used to calculate the logmae objective
    Use as
    ```python
    my_prior_mean = BeamNNPrior(target=torch.tensor(target_beam))  # define prior mean
    gp = SingleTask(X, Y, mean_module=my_prior_mean)   # and construct GP model using it
    ```
    Optional, set the prior mean as fixed and not fit it:
    ```
    for param in custom_mean.mlp.parameters():
        param.requires_grad = False
    ```

    Parameters
    ----------
    target : torch.Tensor
        Target beam in [mu_x, sigma_x, mu_y, sigma_y]
    """

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.mlp = SimpleBeamPredictNN()
        self.mlp.load_state_dict(torch.load(f"nn_for_bo/v2_model_weights.pth"))
        self.mlp.eval()
        self.mlp.double()  # for double input from GPyTorch
        self.target = target

    def forward(self, x):
        logmae = -torch.log(torch.mean(torch.abs(self.mlp(x) - self.target), dim=-1))
        return logmae
