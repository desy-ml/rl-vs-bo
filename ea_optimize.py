from datetime import datetime
from io import BytesIO
import time

import gym
from gym.wrappers import FlattenObservation, RecordVideo, RescaleAction, TimeLimit
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import TD3

from ea_train import ARESEA
from utils import PolishedDonkeyCompatibility, send_to_elog

# import pydoocs
import dummypydoocs as pydoocs


def optimize(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=3.3198e-6,
    target_mu_y_threshold=3.3198e-6,
    target_sigma_x_threshold=3.3198e-6,
    target_sigma_y_threshold=3.3198e-6,
    max_steps=50,
    model_name="polished-donkey-996",
    logbook=False,
    callback=None,
):
    """
    Function used for optimisation during operation.

    Note: Current version only works for polished-donkey-996.
    """
    # config = read_from_yaml(f"models/{model}/config")
    assert model_name == "polished-donkey-996", "Current version only works for polished-donkey-996."
    
    # Load the model
    model = TD3.load(f"models/{model_name}/model")
    
    # Create the environment
    def make_env_polished():
        env = ARESEADOOCS(
            action_mode="delta",
            magnet_init_mode="constant",
            magnet_init_values=np.array([10, -10, 0, 10, 0]),
            reward_mode="differential",
            target_beam_mode="constant",
            target_beam_values=np.array([target_mu_x, target_sigma_x, target_mu_y, target_sigma_y]),
            target_mu_x_threshold=target_mu_x_threshold,
            target_mu_y_threshold=target_mu_y_threshold,
            target_sigma_x_threshold=target_sigma_x_threshold,
            target_sigma_y_threshold=target_sigma_y_threshold,
        )
        if max_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_steps)
        env = ARESEARecorder(env, logbook=logbook, model_name=model_name)
        env = FlattenObservation(env)
        env = PolishedDonkeyCompatibility(env)
        env = RescaleAction(env, -1, 1)
        env = RecordVideo(env, "recordings_function_test")
        
        return env

    env = DummyVecEnv([make_env_polished])
    env = VecNormalize.load(f"models/{model_name}/vec_normalize.pkl", env)
    env.training = False

    # Actual optimisation
    observation = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
    env.close()


class ARESEADOOCS(ARESEA):
    
    def __init__(
        self,
        action_mode="direct",
        magnet_init_mode="zero",
        magnet_init_values=None,
        reward_mode="differential",
        target_beam_mode="random",
        target_beam_values=None,
        target_mu_x_threshold=3.3198e-6,
        target_mu_y_threshold=2.4469e-6,
        target_sigma_x_threshold=3.3198e-6,
        target_sigma_y_threshold=2.4469e-6,
        threshold_hold=1,
        w_mu_x=1.0,
        w_mu_y=1.0,
        w_on_screen=1.0,
        w_sigma_x=1.0,
        w_sigma_y=1.0,
        w_time=1.0,
    ):
        super().__init__(
            action_mode=action_mode,
            magnet_init_mode=magnet_init_mode,
            magnet_init_values=magnet_init_values,
            reward_mode=reward_mode,
            target_beam_mode=target_beam_mode,
            target_beam_values=target_beam_values,
            target_mu_x_threshold=target_mu_x_threshold,
            target_mu_y_threshold=target_mu_y_threshold,
            target_sigma_x_threshold=target_sigma_x_threshold,
            target_sigma_y_threshold=target_sigma_y_threshold,
            threshold_hold=threshold_hold,
            w_mu_x=w_mu_x,
            w_mu_y=w_mu_y,
            w_on_screen=w_on_screen,
            w_sigma_x=w_sigma_x,
            w_sigma_y=w_sigma_y,
            w_time=w_time,
        )

    def is_beam_on_screen(self):
        return True # TODO find better logic

    def get_magnets(self):
        return np.array([
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.RBV")["data"],
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.RBV")["data"],
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.RBV")["data"] / 1000,
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.RBV")["data"],
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.RBV")["data"] / 1000
        ])
    
    def set_magnets(self, magnets):
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.SP", magnets[0])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.SP", magnets[1])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.SP", magnets[2] * 1000)
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.SP", magnets[3])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.SP", magnets[4] * 1000)

        # Wait until magnets have reached their setpoints
        
        time.sleep(3.0) # Wait for magnets to realise they received a command

        magnets = ["AREAMQZM1", "AREAMQZM2", "AREAMCVM1", "AREAMQZM3", "AREAMCHM1"]

        are_busy = [True] * 5
        are_ps_on = [True] * 5
        while any(are_busy) or not all(are_ps_on):
            are_busy = [pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/BUSY")["data"] for magnet in magnets]
            are_ps_on = [pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/PS_ON")["data"] for magnet in magnets]

    def update_accelerator(self):
        self.beam_image = self.capture_clean_beam_image()

    def get_beam_parameters(self):
        img = self.get_beam_image()
        pixel_size = self.get_pixel_size()

        parameters = {}
        for axis, direction in zip([0,1], ["x","y"]):
            projection = img.sum(axis=axis)
            minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
            filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

            half_values, = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2
            else:
                fwhm_pixel = 42     # TODO figure out what to do with these
                center_pixel = 42

            parameters[f"mu_{direction}"] = (center_pixel - len(filtered) / 2) * pixel_size[axis]
            parameters[f"sigma_{direction}"] = fwhm_pixel / 2.355 * pixel_size[axis]
            
        parameters["mu_y"] = -parameters["mu_y"]

        return np.array([
            parameters["mu_x"],
            parameters["sigma_x"],
            parameters["mu_y"],
            parameters["sigma_y"]
        ])

    def get_beam_image(self):
        return self.beam_image

    def get_binning(self):
        return np.array((
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL")["data"],
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL")["data"]
        ))

    def get_screen_resolution(self):
        return np.array([
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/WIDTH")["data"],
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/HEIGHT")["data"]
        ])
    
    def get_pixel_size(self):
        return np.array([
            abs(pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/X.POLY_SCALE")["data"][2]) / 1000,
            abs(pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/Y.POLY_SCALE")["data"][2]) / 1000
        ]) * self.get_binning()

    def capture_clean_beam_image(self, average=5):
        """
        Capture a clean image of the beam from the screen using `average` images with beam on and
        `average` images of the background and then removing the background.
        
        Saves the image to a property of the object.
        """
         # Laser off
        self.set_cathode_laser(False)
        background_images = self.capture_interval(n=average, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        self.set_cathode_laser(True)
        beam_images = self.capture_interval(n=average, dt=0.1)
        median_beam = np.median(beam_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16-1)
        flipped = np.flipud(removed)
        
        return flipped
    
    def capture_interval(self, n, dt):
        """Capture `n` images from the screen and wait `dt` seconds in between them."""
        images = []
        for _ in range(n):
            images.append(self.capture_screen())
            time.sleep(dt)
        return np.array(images)
    
    def capture_screen(self):
        """Capture and image from the screen."""
        return pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ")["data"]

    def set_cathode_laser(self, setto):
        """Sets the bool switch of the cathode laser event to `setto` and waits a second."""
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)


class ARESEARecorder(gym.Wrapper):

    def __init__(self, env, logbook=False, model_name="-", path=None,):
        super().__init__(env)

        self.logbook = logbook
        self.model_name = model_name
        self.path = path

        self.has_previously_run = False

    def reset(self):
        self.finalize_previous_episode()

        observation = super().reset()

        self.observations = [observation]
        self.rewards = []
        self.infos = []
        self.beam_images = [self.env.get_beam_image()]
        self.t_start = datetime.now()
        self.t_end = None
        self.steps_taken = 0
        self.has_previously_run = True

        return observation
    
    def step(self, action):
        observation, reward, done, info = super().step(action)

        self.observations.append(observation)
        self.rewards.append(reward)
        self.infos.append(info)
        self.beam_images.append(self.env.get_beam_image())
        self.steps_taken += 1

        return observation, reward, done, info
    
    def close(self):
        self.finalize_previous_episode()
        return super().close()
    
    def finalize_previous_episode(self):
        if not self.has_previously_run:
            return

        self.t_end = datetime.now()

        if self.path is not None:
            self.write_to_file()
        if self.logbook:
            self.write_to_logbook()
    
    def write_to_file(self):
        pass
    
    def write_to_logbook(self):
        # Create text message
        beam_before = self.observations[0]["beam"]
        beam_after = self.observations[-1]["beam"]
        final_magnets = self.observations[-1]["magnets"]
        msg = f"""Reinforcement learning agent optimised beam on AREABSCR1
        
        Agent: {self.model_name}
        No. of steps: {self.steps_taken}
        Time taken: {self.t_end - self.t_start}

        Beam before:
            mu_x    = {beam_before[0] * 1e3} mm
            sigma_x = {beam_before[1] * 1e3} mm
            mu_y    = {beam_before[2] * 1e3} mm
            sigma_y = {beam_before[3] * 1e3} mm

        Beam after:
            mu_x    = {beam_after[0] * 1e3} mm
            sigma_x = {beam_after[1] * 1e3} mm
            mu_y    = {beam_after[2] * 1e3} mm
            sigma_y = {beam_after[3] * 1e3} mm

        Final magnet settings:
            AREAMQZM1 strength = {final_magnets[0]} 1/m
            AREAMQZM2 strength = {final_magnets[1]} 1/m
            AREAMCVM1 kick     = {final_magnets[2] * 1e3} mrad
            AREAMQZM3 strength = {final_magnets[3]} 1/m
            AREAMCHM1 kick     = {final_magnets[4] * 1e3} mrad
        """

        # Create plot as jpg
        fig, axs = plt.subplots(1, 5, figsize=(30,4))
        self.plot_quadrupole_history(axs[0])
        self.plot_steerer_history(axs[1])
        self.plot_beam_history(axs[2])
        self.plot_beam_before(axs[3])
        self.plot_beam_after(axs[4])
        fig.tight_layout()
        
        buf = BytesIO()
        fig.savefig(buf, dpi=300, format="jpg")
        buf.seek(0)
        img = bytes(buf.read())
    
        # Send to logbook
        send_to_elog(
            elog="areslog",
            author="Autonomous ARES",
            title="RL-based Beam Optimisation on AREABSCR1",
            severity="None",
            text=msg,
            image=img,
        )
        # If logbook not reachable just show everything
        
        print(msg)

    def plot_quadrupole_history(self, ax):
        areamqzm1 = [obs["magnets"][0] for obs in self.observations]
        areamqzm2 = [obs["magnets"][1] for obs in self.observations]
        areamqzm3 = [obs["magnets"][3] for obs in self.observations]

        steps = np.arange(self.steps_taken + 1)

        ax.set_title("Quadrupoles")
        ax.set_xlim([0, self.steps_taken+1])
        ax.set_xlabel("Step")
        ax.set_ylabel("Strength (1/m)")
        ax.plot(steps, areamqzm1, label="AREAMQZM1")
        ax.plot(steps, areamqzm2, label="AREAMQZM2")
        ax.plot(steps, areamqzm3, label="AREAMQZM3")
        ax.legend()
        ax.grid(True)

    def plot_steerer_history(self, ax):
        areamcvm1 = np.array([obs["magnets"][2] for obs in self.observations])
        areamchm2 = np.array([obs["magnets"][4] for obs in self.observations])

        steps = np.arange(self.steps_taken + 1)

        ax.set_title("Steerers")
        ax.set_xlabel("Step")
        ax.set_ylabel("Kick (mrad)")
        ax.set_xlim([0, self.steps_taken+1])
        ax.plot(steps, areamcvm1*1e3, label="AREAMCVM1")
        ax.plot(steps, areamchm2*1e3, label="AREAMCHM2")
        ax.legend()
        ax.grid(True)

    def plot_beam_history(self, ax):
        mu_x = np.array([obs["beam"][0] for obs in self.observations])
        sigma_x = np.array([obs["beam"][1] for obs in self.observations])
        mu_y = np.array([obs["beam"][2] for obs in self.observations])
        sigma_y = np.array([obs["beam"][3] for obs in self.observations])

        target_beam = self.observations[0]["target"]

        steps = np.arange(self.steps_taken + 1)

        ax.set_title("Beam Parameters")
        ax.set_xlim([0, self.steps_taken+1])
        ax.set_xlabel("Step")
        ax.set_ylabel("(mm)")
        ax.plot(steps, mu_x*1e3, label=r"$\mu_x$", c="tab:blue")
        ax.plot(steps, [target_beam[0]*1e3]*len(steps), ls="--", c="tab:blue")
        ax.plot(steps, sigma_x*1e3, label=r"$\sigma_x$", c="tab:orange")
        ax.plot(steps, [target_beam[1]*1e3]*len(steps), ls="--", c="tab:orange")
        ax.plot(steps, mu_y*1e3, label=r"$\mu_y$", c="tab:green")
        ax.plot(steps, [target_beam[2]*1e3]*len(steps), ls="--", c="tab:green")
        ax.plot(steps, sigma_y*1e3, label=r"$\sigma_y$", c="tab:red")
        ax.plot(steps, [target_beam[3]*1e3]*len(steps), ls="--", c="tab:red")
        ax.legend()
        ax.grid(True)
     
    def plot_beam_before(self, ax):
        img = self.beam_images[0]
        screen_size = self.env.get_screen_resolution() * self.env.get_pixel_size()

        ax.set_title("Beam Before (Background Removed)")
        ax.set_xlabel("(mm)")
        ax.set_ylabel("(mm)")
        ax.imshow(
            img,
            vmin=0,
            aspect="equal",
            interpolation="none",
            extent=(
                -screen_size[0] / 2 * 1e3,
                screen_size[0] / 2 * 1e3,
                -screen_size[1] / 2 * 1e3,
                screen_size[1] / 2 * 1e3,
            ),
        )

    def plot_beam_after(self, ax):
        img = self.beam_images[-1]
        screen_size = self.env.get_screen_resolution() * self.env.get_pixel_size()

        ax.set_title("Beam After (Background Removed)")
        ax.set_xlabel("(mm)")
        ax.set_ylabel("(mm)")
        ax.imshow(
            img,
            vmin=0,
            aspect="equal",
            interpolation="none",
            extent=(
                -screen_size[0] / 2 * 1e3,
                screen_size[0] / 2 * 1e3,
                -screen_size[1] / 2 * 1e3,
                screen_size[1] / 2 * 1e3,
            ),
        )
