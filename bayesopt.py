from typing import Optional, Union

import gym
import numpy as np
import torch
import torch.nn as nn
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
)
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.optim import optimize_acqf
from gpytorch.means.mean import Mean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gym.spaces.utils import unflatten

# TODO Just for testing, we can also add proximal biasing?


def scale_action(env, observation, filter_action=None):
    """Scale the observed magnet settings to proper action values"""
    unflattened = (
        unflatten(env.unwrapped.observation_space, observation)
        if not isinstance(observation, dict)
        else observation
    )
    magnet_values = unflattened["magnets"]
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
    beta=0.2,
    fixparam: Optional[dict] = None,
    mean_module: Optional[Mean] = None,
    outcome_transform: Optional[OutcomeTransform] = Standardize(m=1),
):
    """
    Suggest Next Sample for BO
    """
    gp = SingleTaskGP(
        X, Y, mean_module=mean_module, outcome_transform=outcome_transform
    )
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
    elif acquisition == "PI":
        acq = ProbabilityOfImprovement(model=gp, best_f=best_y)
    elif acquisition == "UCB":
        acq = UpperConfidenceBound(gp, beta=beta)

    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=n_points,
        num_restarts=10,
        raw_samples=256,
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
    w_on_screen=10,
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
        _, objective, done, _ = env.step(action)
        Y[i] = torch.tensor(objective)

    # In the loop
    for i in range(budget):
        action = get_next_samples(X, Y, Y.max(), bounds, n_points=1, acquisition=acq)
        _, objective, done, _ = env.step(action.detach().numpy().flatten())
        # append data
        X = torch.cat([X, action])
        Y = torch.cat([Y, torch.tensor([[objective]], dtype=torch.float32)])

        if done:
            print(f"Optimization succeeds in {i} steps")
            break

    return X.detach().numpy(), Y.detach().numpy()


# Use a NN as GP prior for BO
class SimpleBeamPredictNN(nn.Module):
    """
    A simple FCNN to predict the output beam parameters assuming centered incoming beam.
    """

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


class SimpleBeamPredictNNV3(nn.Module):
    def __init__(self, include_bias=False) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 4, bias=include_bias),
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
    w_on_screen: float (optional)
        Weight of the reward given for beam being on/off screen
    """

    def __init__(
        self,
        target: torch.Tensor,
        w_on_screen: float = 10,
        screen_resolution=(2448, 2040),
        screen_pixel_size=(3.3198e-6, 2.4469e-6),
    ):
        super().__init__()
        self.mlp = SimpleBeamPredictNNV3()
        self.mlp.load_state_dict(torch.load("nn_for_bo/v3_model_weights.pth"))
        self.mlp.eval()
        self.mlp.double()  # for double input from GPyTorch
        self.target = target
        # Screen Size calculation for on screen reward
        self.w_on_screen_reward = w_on_screen
        self.half_x_size = screen_resolution[0] * screen_pixel_size[0] / 2
        self.half_y_size = screen_resolution[1] * screen_pixel_size[1] / 2

        # additional scaling and shift
        self.register_parameter(
            name="out_weight", parameter=torch.nn.Parameter(torch.tensor([0.0]))
        )
        self.register_parameter(
            name="out_bias", parameter=torch.nn.Parameter(torch.tensor([0.0]))
        )

    def forward(self, x):
        out_beam = self.mlp(x)
        out_beam = denormalize_output(out_beam)
        logmae = -torch.log(torch.mean(torch.abs(out_beam - self.target), dim=-1))

        is_beam_on_screen = torch.logical_and(
            torch.abs(out_beam[..., 0]) < self.half_x_size,
            torch.abs(out_beam[..., 2]) < self.half_y_size,
        )  # check if both x and y position are inside the screen
        is_beam_on_screen = torch.where(is_beam_on_screen, 1, -1)

        on_screen_reward = self.w_on_screen_reward * is_beam_on_screen
        pred_reward = logmae + on_screen_reward
        pred_reward = (
            pred_reward * torch.nn.functional.softplus(self.out_weight) + self.out_bias
        )
        return pred_reward


def denormalize_output(y_nn: torch.Tensor) -> torch.Tensor:
    y_nn = (y_nn + torch.tensor([0, 1, 0, 1])) * 0.005  # denormalize to 5 mm
    return y_nn


class BayesianOptimizationAgent:
    """
    Provide an interface to Bayesian Optimisation similar to Stable Baselines3 RL
    agents.
    """

    def __init__(
        self,
        env: gym.Env,
        filter_action=None,
        stepsize=0.1,
        init_samples=5,
        acquisition="EI",
        mean_module=None,
    ) -> None:
        self.env = env
        self.filter_action = filter_action
        self.stepsize = stepsize
        self.init_samples = init_samples
        self.acquisition = acquisition
        self.mean_module = mean_module

    def predict(self, observation, reward=None):
        self.validate_x_and_y_state()

        # If a reward was passed, create Y or append to Y depending on if Y exists
        if reward is not None:
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
            self.Y = (
                torch.cat([self.Y, reward_tensor])
                if hasattr(self, "Y")
                else reward_tensor
            )

        # First sample
        if not hasattr(self, "X"):
            initial_action = scale_action(self.env, observation, self.filter_action)
            self.X = torch.tensor([initial_action], dtype=torch.float32)
            return initial_action

        # Initial random samples after initial sample
        if len(self.X) < self.init_samples:
            last_action = self.X[0].detach().numpy()
            bounds = get_new_bound(self.env, last_action, self.stepsize)
            new_action = np.random.uniform(low=bounds[0], high=bounds[1])
            new_action_tensor = torch.tensor(new_action, dtype=torch.float32).reshape(
                1, -1
            )
            self.X = torch.cat([self.X, new_action_tensor])
            return new_action

        # All "normal" samples after the initial samples
        last_action = self.X[-1].detach().numpy()
        bounds = get_new_bound(self.env, last_action, self.stepsize)
        action_tensor = get_next_samples(
            self.X.double(),
            self.Y.double(),
            self.Y.max(),
            torch.tensor(bounds, dtype=torch.double),
            n_points=1,
            acquisition=self.acquisition,
            mean_module=self.mean_module,
        )
        self.X = torch.cat([self.X, action_tensor])

        return self.X[-1].detach().numpy()

    def validate_x_and_y_state(self) -> None:
        """
        Raise `AssertionError` when `self.X` and `self.Y` are in an invalid state in
        terms of their existance and shapes.
        """
        no_x_and_y = not hasattr(self, "X") and not hasattr(self, "Y")
        only_x = hasattr(self, "X") and len(self.X) == 1 and not hasattr(self, "Y")
        both_x_and_y = (
            hasattr(self, "X") and hasattr(self, "Y") and len(self.X) - len(self.Y) == 1
        )
        assert no_x_and_y or only_x or both_x_and_y, (
            f"BO optimisation has reach invalid state {no_x_and_y = }, {only_x = },"
            f" {both_x_and_y = }"
        )
