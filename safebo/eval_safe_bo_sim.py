import os
from pathlib import Path
from time import sleep

import numpy as np
from gym.wrappers import RescaleAction, TimeLimit

from backend import EACheetahBackend
from environment import EATransverseTuning
from trial import Trial, load_trials
from utils import RecordEpisode


def try_problem(trial_index: int, trial: Trial) -> None:
    # Create the environment
    cheetah_backend = EACheetahBackend(
        incoming_mode="constant",
        incoming_values=trial.incoming_beam,
        misalignment_mode="constant",
        misalignment_values=trial.misalignments,
    )
    env = EATransverseTuning(
        backend=cheetah_backend,
        action_mode="direct_unidirectional_quads",
        log_beam_distance=True,
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        reward_mode="feedback",
        target_beam_mode="constant",
        target_beam_values=trial.target_beam,
        target_mu_x_threshold=None,
        target_mu_y_threshold=None,
        target_sigma_x_threshold=None,
        target_sigma_y_threshold=None,
        threshold_hold=5,
        w_done=0.0,
        w_mu_x=1.0,
        w_mu_x_in_threshold=0.0,
        w_mu_y=1.0,
        w_mu_y_in_threshold=0.0,
        w_on_screen=5.0,
        w_sigma_x=1.0,
        w_sigma_x_in_threshold=0.0,
        w_sigma_y=1.0,
        w_sigma_y_in_threshold=0.0,
        w_time=0.0,
        normalize_beam_distance=False,
    )
    env = TimeLimit(env, 150)
    env = RecordEpisode(
        env, save_dir=f"data/bo_vs_rl/simulation/safe_bo/problem_{trial_index:03d}"
    )
    env = RescaleAction(env, -1, 1)

    safe_bo = MatlabSafeBO()

    # Actual optimisation
    _ = env.reset()
    done = False
    while not done:
        try:
            action = safe_bo.request_sample()
        except MatlabSafeBO.OptimumReachedException:
            done = True
            break

        _, reward, done, _ = env.step(action)
        safe_bo.report_objective(-reward)  # NOTE minimisation?

    # Return to best seen sample
    set_to_best = False
    if set_to_best:
        action = safe_bo.request_best()
        _ = env.step(action)

    env.close()
    del safe_bo


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
    OPTIMUM_REACHED_FILE = f"{PREFIX}_optimum_reached"

    class OptimumReachedException(Exception):
        pass

    def __init__(self) -> None:
        # os.popen("matlab -nosplash -nodesktop -nojvm -r start_matlab")

        self.last_communication = "none"

    def __del__(self) -> None:
        # Tell Matlab to shut down by placing a file
        with open(self.SHUTDOWN_FILE, "w") as f:
            f.write(" ")

        # Wait until shutdown file has disappeared, i.e. Matlab acknoledged the request
        # and shut down
        sleep(60.0)

        os.remove(self.SHUTDOWN_FILE)

    def request_sample(self) -> np.ndarray:
        """Request the position of the next sample from Matlab."""
        assert self.last_communication in ["none", "objective", "best"]

        # Ask for sample by placing file
        with open(self.REQUEST_SAMPLE_FILE, "w") as f:
            f.write(" ")

        # Wait for the sample answer file to exist and then read it
        print("Wait for the sample answer file to exist and then read it")
        while os.path.exists(self.REQUEST_SAMPLE_FILE) or not os.path.exists(
            self.SAMPLE_FILE
        ):
            if os.path.exists(self.OPTIMUM_REACHED_FILE):
                os.remove(self.REQUEST_SAMPLE_FILE)
                os.remove(self.OPTIMUM_REACHED_FILE)
                raise self.OptimumReachedException()
            sleep(1.0)
        with open(self.SAMPLE_FILE, "r") as f:
            sample = f.read()
        sample = sample.split(",")[:-1]
        sample = [float(x) for x in sample]
        sample = np.array(sample)

        os.remove(self.SAMPLE_FILE)

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
        while os.path.exists(self.REQUEST_BEST_FILE) or not os.path.exists(
            self.BEST_FILE
        ):
            sleep(1.0)
        with open(self.BEST_FILE, "r") as f:
            sample = f.read()
        sample = sample.split(",")
        sample = [float(x) for x in sample]
        sample = np.array(sample)

        os.remove(self.BEST_FILE)

        self.last_communication = "best"

        print(f"Got {sample = }")

        return sample


def main():
    trials = load_trials(Path("trials.yaml"))

    i = 34
    try_problem(i, trials[i])


if __name__ == "__main__":
    main()
