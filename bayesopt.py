from typing import Optional, Union

import gym
import numpy as np
import torch
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


def observation_to_scaled_action(env, observation, filter_action=[0, 1, 2, 3, 4]):
    """
    Extract from the unscaled observation the magnet settings and scale them as a
    correct input to the `RescaleAction` wrapper.
    """
    magnets = observation["magnets"]
    filtered_magnets = magnets[filter_action].squeeze()

    min_action = env.action_space.low
    max_action = env.action_space.high
    low = env.unwrapped.action_space.low
    high = env.unwrapped.action_space.high

    action = min_action + (max_action - min_action) * (filtered_magnets - low) / (
        high - low
    )

    return action


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
        beta=0.2,
    ) -> None:
        self.env = env
        self.filter_action = filter_action
        self.stepsize = stepsize
        self.init_samples = init_samples
        self.acquisition = acquisition
        self.mean_module = mean_module
        self.beta = beta

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
            initial_action = observation_to_scaled_action(
                self.env, observation, self.filter_action
            )
            self.X = torch.tensor(initial_action, dtype=torch.float32).reshape(1, -1)
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
            beta=self.beta,
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
