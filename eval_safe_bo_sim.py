import json
import os
from time import sleep

import numpy as np
from gym.wrappers import RescaleAction, TimeLimit
from tqdm.notebook import tqdm

from ea_train import ARESEACheetah
from utils import RecordEpisode


def convert_incoming_from_problem(problem: dict) -> np.ndarray:
    return np.array(
        [
            problem["incoming"]["energy"],
            problem["incoming"]["mu_x"],
            problem["incoming"]["mu_xp"],
            problem["incoming"]["mu_y"],
            problem["incoming"]["mu_yp"],
            problem["incoming"]["sigma_x"],
            problem["incoming"]["sigma_xp"],
            problem["incoming"]["sigma_y"],
            problem["incoming"]["sigma_yp"],
            problem["incoming"]["sigma_s"],
            problem["incoming"]["sigma_p"],
        ]
    )


def convert_misalignments_from_problem(problem: dict) -> np.ndarray:
    return np.array(problem["misalignments"])


def convert_target_from_problem(problem: dict) -> np.ndarray:
    return np.array(
        [
            problem["desired"][0],
            problem["desired"][2],
            problem["desired"][1],
            problem["desired"][3],
        ]
    )


def try_problem(problem_index: int, problem: dict):
    # Create the environment
    env = ARESEACheetah(
        action_mode="direct_unidirectional_quads",
        incoming_mode="constant",
        incoming_values=convert_incoming_from_problem(problem),
        log_beam_distance=True,
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        misalignment_mode="constant",
        misalignment_values=convert_misalignments_from_problem(problem),
        reward_mode="feedback",
        target_beam_mode="constant",
        target_beam_values=convert_target_from_problem(problem),
        target_mu_x_threshold=None,
        target_mu_y_threshold=None,
        target_sigma_x_threshold=None,
        target_sigma_y_threshold=None,
        threshold_hold=5,
        w_done=0.0,
        w_mu_x=1.1,
        w_mu_x_in_threshold=0.0,
        w_mu_y=1.0,
        w_mu_y_in_threshold=0.0,
        w_on_screen=10.0,
        w_sigma_x=1.0,
        w_sigma_x_in_threshold=0.0,
        w_sigma_y=1.0,
        w_sigma_y_in_threshold=0.0,
        w_time=0.0,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(
        env, save_dir=f"data/bo_vs_rl/simulation/safe_bo/problem_{problem_index:03d}"
    )
    env = RescaleAction(env, -1, 1)

    safe_bo = MatlabSafeBO()

    # Actual optimisation
    _ = env.reset()
    done = False
    while not done:
        action = safe_bo.request_sample()
        _, reward, done, _ = env.step(action)
        safe_bo.report_objective(reward)

    # Return to best seen sample
    set_to_best = True
    if set_to_best:
        action = safe_bo.request_best()
        _ = env.step(action)

    env.close()


class MatlabSafeBO:
    """
    Interface class to safe BO implementation in Matlab.

    This class expects that on the Matlab side there exists a function that takes a
    sample in the form of five normalised magnet values between -1 and 1, and returns
    and objective value.
    This function should look something like this:
    ```
    Wait for  MSBO_request file to appear
    Delete MSBO_request file
    Write MSBO_sample file of five comma-sperated values
    Wait for MSBO_sample file to be deleted and MSBO_objective file to appear
    Read objective value from MSBO_objective file and delete the file
    Return objective value
    ```

    The Matlab script may shut down if it sees the `MSBO_good_night` file. It must
    delete this file before shutting down.

    By the time the optimisation is done, non of the files must be left.

    TODO Somehow include the request for best in here (not immediately needed for
    simulation evaluation.)
    """

    PREFIX = "MSBO"

    BEST_FILE = f"{PREFIX}_best"
    OBJECTIVE_FILE = f"{PREFIX}_objective"
    REQUEST_BEST_FILE = f"{PREFIX}_best_request"
    REQUEST_SAMPLE_FILE = f"{PREFIX}_sample_request"
    SAMPLE_FILE = f"{PREFIX}_sample"
    SHUTDOWN_FILE = f"{PREFIX}_good_night"

    def __init__(self) -> None:
        # os.popen("matlab", "safe_bo.m")  # TODO Is this correct?

        self.last_communication = "none"

    def __del__(self) -> None:
        # Tell Matlab to shut down by placing a file
        with open(self.SHUTDOWN_FILE, "w") as f:
            f.write(" ")

        # Wait until shutdown file has disappeared, i.e. Matlab acknoledged the request
        # and shut down
        while os.path.exists(self.SHUTDOWN_FILE):
            sleep(1.0)

    def request_sample(self) -> np.ndarray:
        """Request the position of the next sample from Matlab."""
        assert self.last_communication in ["none", "objective", "best"]

        # Ask for sample by placing file
        with open(self.REQUEST_SAMPLE_FILE, "w") as f:
            f.write(" ")

        # Wait for the sample answer file to exist and then read it
        print("Wait for the sample answer file to exist and then read it")
        while os.path.exists(self.REQUEST_SAMPLE_FILE) and not os.path.exists(
            self.SAMPLE_FILE
        ):
            sleep(1.0)
        with open(self.SAMPLE_FILE, "r") as f:
            sample = f.read()
        sample = sample.split(",")
        sample = [float(x) for x in sample]
        sample = np.array(sample)

        self.last_communication = "sample"

        print(f"Got {sample = }")

        return sample

    def report_objective(self, objective: float) -> None:
        """Report the objective of the last sample to Matlab."""
        assert self.last_communication == "sample"

        print(f"Reporting {objective = }")

        # Write file with objective value
        objective_string = str(objective)
        with open(self.OBJECTIVE_FILE, "w") as f:
            f.write(objective_string)

        # Wait until Matlab has picked up objective value file
        while os.path.exists(self.OBJECTIVE_FILE):
            sleep(1.0)

        self.last_communication = "objective"

    def request_best(self) -> np.ndarray:
        """Request the position of the best sample from Matlab."""
        assert self.last_communication in ["objective", "best"]

        # Ask for best sample by placing file
        with open(self.REQUEST_BEST_FILE, "w") as f:
            f.write(" ")

        # Wait for the sample answer file to exist and then read it
        print("Wait for the best answer file to exist and then read it")
        while os.path.exists(self.REQUEST_BEST_FILE) and not os.path.exists(
            self.BEST_FILE
        ):
            sleep(1.0)
        with open(self.BEST_FILE, "r") as f:
            sample = f.read()
        sample = sample.split(",")
        sample = [float(x) for x in sample]
        sample = np.array(sample)

        self.last_communication = "best"

        print(f"Got {sample = }")

        return sample


def main():
    with open("problems.json", "r") as f:
        problems = json.load(f)

    for i, problem in tqdm(enumerate(problems)):
        try_problem(i, problem)


if __name__ == "__main__":
    main()
