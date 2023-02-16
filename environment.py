from functools import partial
from typing import Optional

import cv2
import gym
import numpy as np
from gym import spaces

from backend import TransverseTuningBaseBackend


class TransverseTuningEnv(gym.Env):
    """
    Base environment for position and focusing an electron beam on a diagnostic screen
    using quadrupole and steering magnets.
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 2}

    def reset(self) -> np.ndarray:
        self.backend.reset()

        if self.magnet_init_mode == "constant":
            self.backend.set_magnets(self.magnet_init_values)
        elif self.magnet_init_mode == "random":
            self.backend.set_magnets(self.observation_space["magnets"].sample())
        elif self.magnet_init_mode is None:
            pass  # This really is intended to do nothing
        else:
            raise ValueError(
                f'Invalid value "{self.magnet_init_mode}" for magnet_init_mode'
            )

        if self.target_beam_mode == "constant":
            self.target_beam = self.target_beam_values
        elif self.target_beam_mode == "random":
            self.target_beam = self.observation_space["target"].sample()
        else:
            raise ValueError(
                f'Invalid value "{self.target_beam_mode}" for target_beam_mode'
            )

        # Update anything in the accelerator (mainly for running simulations)
        self.backend.update()

        self.initial_screen_beam = self.backend.get_beam_parameters()
        self.previous_beam = self.initial_screen_beam
        self.is_in_threshold_history = []
        self.steps_taken = 0

        observation = {
            "beam": self.initial_screen_beam.astype("float32"),
            "magnets": self.backend.get_magnets().astype("float32"),
            "target": self.target_beam.astype("float32"),
        }

        return observation

    def step(self, action: np.ndarray) -> tuple:
        self.take_action(action)

        # Run the simulation
        self.backend.update()

        current_beam = self.backend.get_beam_parameters()
        self.steps_taken += 1

        # Build observation
        observation = {
            "beam": current_beam.astype("float32"),
            "magnets": self.backend.get_magnets().astype("float32"),
            "target": self.target_beam.astype("float32"),
        }

        # For readibility in computations below
        cb = current_beam
        tb = self.target_beam

        # Compute if done (beam within threshold for a certain time)
        threshold = np.array(
            [
                self.target_mu_x_threshold,
                self.target_sigma_x_threshold,
                self.target_mu_y_threshold,
                self.target_sigma_y_threshold,
            ],
            dtype=np.double,
        )
        threshold = np.nan_to_num(threshold)
        is_in_threshold = np.abs(cb - tb) < threshold
        self.is_in_threshold_history.append(is_in_threshold)
        is_stable_in_threshold = bool(
            np.array(self.is_in_threshold_history[-self.threshold_hold :]).all()
        )
        done = is_stable_in_threshold and len(self.is_in_threshold_history) > 5

        # Compute reward
        on_screen_reward = 1 if self.backend.is_beam_on_screen() else -1
        time_reward = -1
        done_reward = int(done)
        beam_reward = self.compute_beam_reward(current_beam)

        reward = 0
        reward += self.w_on_screen * on_screen_reward
        reward += self.w_beam * beam_reward
        reward += self.w_time * time_reward
        reward += self.w_mu_x_in_threshold * is_in_threshold[0]
        reward += self.w_sigma_x_in_threshold * is_in_threshold[1]
        reward += self.w_mu_y_in_threshold * is_in_threshold[2]
        reward += self.w_sigma_y_in_threshold * is_in_threshold[3]
        reward += self.w_done * done_reward
        reward = float(reward)

        # Put together info
        info = {
            "binning": self.backend.get_binning(),
            "l1_distance": self.compute_beam_distance(current_beam, ord=1),
            "on_screen_reward": on_screen_reward,
            "pixel_size": self.backend.get_pixel_size(),
            "screen_resolution": self.backend.get_screen_resolution(),
            "time_reward": time_reward,
        }
        info.update(self.backend.get_info())

        self.previous_beam = current_beam

        return observation, reward, done, info

    def take_action(self, action: np.ndarray) -> None:
        """Take `action` according to the environment's configuration."""
        if self.action_mode == "direct":
            self.backend.set_magnets(action)
        elif self.action_mode == "delta":
            magnet_values = self.backend.get_magnets()
            self.backend.set_magnets(magnet_values + action)
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

    def compute_beam_reward(self, current_beam: np.ndarray) -> float:
        """Compute reward about the current beam's difference to the target beam."""
        compute_beam_distance = partial(
            self.compute_beam_distance, ord=self.beam_distance_ord
        )

        # TODO I'm not sure if the order with log is okay this way

        if self.logarithmic_beam_distance:
            compute_raw_beam_distance = compute_beam_distance
            compute_beam_distance = lambda beam: np.log(  # noqa: E731
                compute_raw_beam_distance(beam)
            )

        if self.reward_mode == "feedback":
            current_distance = compute_beam_distance(current_beam)
            beam_reward = -current_distance
        elif self.reward_mode == "differential":
            current_distance = compute_beam_distance(current_beam)
            previous_distance = compute_beam_distance(self.previous_beam)
            beam_reward = previous_distance - current_distance
        else:
            raise ValueError(f"Invalid value '{self.reward_mode}' for reward_mode")

        if self.normalize_beam_distance:
            initial_distance = compute_beam_distance(self.initial_screen_beam)
            beam_reward /= initial_distance

        return beam_reward

    def compute_beam_distance(self, beam: np.ndarray, ord: int = 2) -> float:
        """
        Compute distance of `beam` to `self.target_beam`. Eeach beam parameter is
        weighted by its configured weight.
        """
        weights = np.array([self.w_mu_x, self.w_sigma_x, self.w_mu_y, self.w_sigma_y])
        weighted_current = weights * beam
        weighted_target = weights * self.target_beam
        return float(np.linalg.norm(weighted_target - weighted_current, ord=ord))


class EATransverseTuning(TransverseTuningEnv):
    """
    Environment for positioning and focusing the beam on AREABSCR1 using AREAMQZM1,
    AREAMQZM2, AREAMCVM1, AREAMQZM3 and AREAMCHM1.

    :param backend: Backend for communication with either a simulation or the control
        system
    :param action_mode: Choose weather actions set magnet settings directly (`"direct"`)
        or change magnet settings (`"delta"`).
    :param beam_distance_ord: Order of distance to use to compute distance between
        current beam and target beam.
    :param logarithmic_beam_distance: Whether to take the logarithm of the beam
        distance.
    :param magnet_init_mode: Magnet initialisation on `reset`. Set to `None` for magnets
        to stay at their current settings, `"random"` to be set to random settings or
        `"constant"` to set them to the settings given by `magnet_init_values`.
    :param magnet_init_values: Values to set magnets to on `reset`. Is only used when
        `magnet_init_mode` is set to `"constant"`.
    :param max_quad_delta: Limit of by how much quadrupole settings may be changed when
        `action_mode` is set to `"delta"`.
    :param max_steerer_delta: Limit of by how much steerer settings may be changed when
        `action_mode` is set to `"delta"`.
    :param normalize_beam_distance: Whether to divide beam distance by the initial beam
        distance in order to scale it to a value around 1.
    :param reward_mode: Set to `"feedback"` to compute the reward as the negative beam
        distance. Set to `"differential"` to compute the reward as the reduction of the
        beam distance.
    :param target_beam_mode: Setting of target beam on `reset`. Choose from `"constant"`
        or `"random"`. The `"constant"` setting requires `target_beam_values` to be set.
    :param target_beam_values: Target beam to use when `target_beam_mode` is set to
        `"constant"`.
    :param target_mu_x_threshold: Distance from target mu_x below which mu_x is
        considered to be achieved.
    :param target_mu_y_threshold: Distance from target mu_y below which mu_y is
        considered to be achieved.
    :param target_sigma_x_threshold: Distance from target sigma_x below which sigma_x is
        considered to be achieved.
    :param target_sigma_y_threshold: Distance from target sigma_y below which sigma_y is
        considered to be achieved.
    :param threshold_hold: Number of steps that all beam parameters difference must be
        below their thresolds before an episode is terminated as successful.
    :param w_beam: Weight of all beam parameter-related rewards in the total reward.
    :param w_done: Weight of the successful episode termination bonus in the total
        reward.
    :param w_mu_x: Weight of the mu_x component in the beam parameter-related reward.
    :param w_mu_x_in_threshold: Weight of the bonus reward when mu_x is within its
        threshold from the target.
    :param w_mu_y: Weight of the mu_y component in the beam parameter-related reward.
    :param w_mu_y_in_threshold: Weight of the bonus reward when mu_y is within its
        threshold from the target.
    :param w_sigma_x: Weight of the sigma_x component in the beam parameter-related
        reward.
    :param w_sigma_x_in_threshold: Weight of the bonus reward when sigma_x is within its
        threshold from the target.
    :param w_sigma_y: Weight of the sigma_y component in the beam parameter-related
        reward.
    :param w_sigma_y_in_threshold: Weight of the bonus reward when sigma_y is within its
        threshold from the target.
    :param w_time: Weight of the reward received for each passing time step.
    """

    def __init__(
        self,
        backend: TransverseTuningBaseBackend,
        action_mode: str = "direct",
        beam_distance_ord: int = 1,
        logarithmic_beam_distance: bool = False,
        magnet_init_mode: Optional[str] = None,
        magnet_init_values: Optional[np.ndarray] = None,
        max_quad_delta: Optional[float] = None,
        max_steerer_delta: Optional[float] = None,
        normalize_beam_distance: bool = True,
        reward_mode: str = "differential",
        target_beam_mode: str = "random",
        target_beam_values: Optional[np.ndarray] = None,
        target_mu_x_threshold: float = 3.3198e-6,
        target_mu_y_threshold: float = 2.4469e-6,
        target_sigma_x_threshold: float = 3.3198e-6,
        target_sigma_y_threshold: float = 2.4469e-6,
        threshold_hold: int = 1,
        unidirectional_quads: bool = False,
        w_beam: float = 0.0,
        w_done: float = 0.0,
        w_mu_x: float = 0.0,
        w_mu_x_in_threshold: float = 0.0,
        w_mu_y: float = 0.0,
        w_mu_y_in_threshold: float = 0.0,
        w_on_screen: float = 0.0,
        w_sigma_x: float = 0.0,
        w_sigma_x_in_threshold: float = 0.0,
        w_sigma_y: float = 0.0,
        w_sigma_y_in_threshold: float = 0.0,
        w_time: float = 0.0,
    ) -> None:
        self.backend = backend

        self.action_mode = action_mode
        self.beam_distance_ord = beam_distance_ord
        self.logarithmic_beam_distance = logarithmic_beam_distance
        self.magnet_init_mode = magnet_init_mode
        self.magnet_init_values = magnet_init_values
        self.max_quad_delta = max_quad_delta
        self.max_steerer_delta = max_steerer_delta
        self.normalize_beam_distance = normalize_beam_distance
        self.reward_mode = reward_mode
        self.target_beam_mode = target_beam_mode
        self.target_beam_values = target_beam_values
        self.target_mu_x_threshold = target_mu_x_threshold
        self.target_mu_y_threshold = target_mu_y_threshold
        self.target_sigma_x_threshold = target_sigma_x_threshold
        self.target_sigma_y_threshold = target_sigma_y_threshold
        self.threshold_hold = threshold_hold
        self.unidirectional_quads = unidirectional_quads
        self.w_beam = w_beam
        self.w_done = w_done
        self.w_mu_x = w_mu_x
        self.w_mu_x_in_threshold = w_mu_x_in_threshold
        self.w_mu_y = w_mu_y
        self.w_mu_y_in_threshold = w_mu_y_in_threshold
        self.w_on_screen = w_on_screen
        self.w_sigma_x = w_sigma_x
        self.w_sigma_x_in_threshold = w_sigma_x_in_threshold
        self.w_sigma_y = w_sigma_y
        self.w_sigma_y_in_threshold = w_sigma_y_in_threshold
        self.w_time = w_time

        if unidirectional_quads:
            self.magnet_space = spaces.Box(
                low=np.array([0, -72, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 0, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            )
        else:
            self.magnet_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            )

        # Create action space
        if self.action_mode == "direct":
            self.action_space = self.magnet_space
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=np.array(
                    [
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                        -self.max_steerer_delta,
                        -self.max_quad_delta,
                        -self.max_steerer_delta,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        self.max_quad_delta,
                        self.max_quad_delta,
                        self.max_steerer_delta,
                        self.max_quad_delta,
                        self.max_steerer_delta,
                    ],
                    dtype=np.float32,
                ),
            )
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

        # Create observation space
        self.observation_space = spaces.Dict(
            {
                "beam": spaces.Box(
                    low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                ),
                "magnets": self.magnet_space,
                "target": spaces.Box(
                    low=np.array([-2e-3, 0, -2e-3, 0], dtype=np.float32),
                    high=np.array([2e-3, 2e-3, 2e-3, 2e-3], dtype=np.float32),
                ),
            }
        )

        # Setup the accelerator (either simulation or the actual machine)
        self.backend.setup()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        assert mode == "rgb_array" or mode == "human"

        binning = self.backend.get_binning()
        pixel_size = self.backend.get_pixel_size()
        resolution = self.backend.get_screen_resolution()

        # Read screen image and make 8-bit RGB
        img = self.backend.get_screen_image()
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)

        # Redraw beam image as if it were binning = 4
        render_resolution = (resolution * binning / 4).astype("int")
        img = cv2.resize(img, render_resolution)

        # Draw desired ellipse
        tb = self.target_beam
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(tb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(tb[1] / pixel_size_b4[0])
        e_pos_y = int(-tb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(tb[3] / pixel_size_b4[1])
        blue = (255, 204, 79)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, blue, 2
        )

        # Draw beam ellipse
        cb = self.backend.get_beam_parameters()
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(cb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(cb[1] / pixel_size_b4[0])
        e_pos_y = int(-cb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(cb[3] / pixel_size_b4[1])
        red = (0, 0, 255)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, red, 2
        )

        # Adjust aspect ratio
        new_width = int(img.shape[1] * pixel_size_b4[0] / pixel_size_b4[1])
        img = cv2.resize(img, (new_width, img.shape[0]))

        # Add magnet values and beam parameters
        magnets = self.backend.get_magnets()
        padding = np.full(
            (int(img.shape[0] * 0.27), img.shape[1], 3), fill_value=255, dtype=np.uint8
        )
        img = np.vstack([img, padding])
        black = (0, 0, 0)
        red = (0, 0, 255)
        green = (0, 255, 0)
        img = cv2.putText(
            img, f"Q1={magnets[0]:.2f}", (15, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img, f"Q2={magnets[1]:.2f}", (215, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CV={magnets[2]*1e3:.2f}",
            (415, 545),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        img = cv2.putText(
            img, f"Q3={magnets[3]:.2f}", (615, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CH={magnets[4]*1e3:.2f}",
            (15, 585),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        mu_x_color = black
        if self.target_mu_x_threshold != np.inf:
            mu_x_color = (
                green if abs(cb[0] - tb[0]) < self.target_mu_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"mx={cb[0]*1e3:.2f}",
            (15, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_x_color,
        )
        sigma_x_color = black
        if self.target_sigma_x_threshold != np.inf:
            sigma_x_color = (
                green if abs(cb[1] - tb[1]) < self.target_sigma_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"sx={cb[1]*1e3:.2f}",
            (215, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_x_color,
        )
        mu_y_color = black
        if self.target_mu_y_threshold != np.inf:
            mu_y_color = (
                green if abs(cb[2] - tb[2]) < self.target_mu_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"my={cb[2]*1e3:.2f}",
            (415, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_y_color,
        )
        sigma_y_color = black
        if self.target_sigma_y_threshold != np.inf:
            sigma_y_color = (
                green if abs(cb[3] - tb[3]) < self.target_sigma_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"sy={cb[3]*1e3:.2f}",
            (615, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_y_color,
        )

        if mode == "human":
            cv2.imshow("ARES EA", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class BCTransverseTuning(TransverseTuningEnv):
    """
    Environment for positioning and focusing the beam on ARBCBSCE1 using ARMRMQZM4,
    ARMRMQZM5, ARMRMCVM5, ARMRMCHM5 and ARMRMQZM6.

    :param backend: Backend for communication with either a simulation or the control
        system
    :param action_mode: Choose weather actions set magnet settings directly (`"direct"`)
        or change magnet settings (`"delta"`).
    :param beam_distance_ord: Order of distance to use to compute distance between
        current beam and target beam.
    :param logarithmic_beam_distance: Whether to take the logarithm of the beam
        distance.
    :param magnet_init_mode: Magnet initialisation on `reset`. Set to `None` for magnets
        to stay at their current settings, `"random"` to be set to random settings or
        `"constant"` to set them to the settings given by `magnet_init_values`.
    :param magnet_init_values: Values to set magnets to on `reset`. Is only used when
        `magnet_init_mode` is set to `"constant"`.
    :param max_quad_delta: Limit of by how much quadrupole settings may be changed when
        `action_mode` is set to `"delta"`.
    :param max_steerer_delta: Limit of by how much steerer settings may be changed when
        `action_mode` is set to `"delta"`.
    :param normalize_beam_distance: Whether to divide beam distance by the initial beam
        distance in order to scale it to a value around 1.
    :param reward_mode: Set to `"feedback"` to compute the reward as the negative beam
        distance. Set to `"differential"` to compute the reward as the reduction of the
        beam distance.
    :param target_beam_mode: Setting of target beam on `reset`. Choose from `"constant"`
        or `"random"`. The `"constant"` setting requires `target_beam_values` to be set.
    :param target_beam_values: Target beam to use when `target_beam_mode` is set to
        `"constant"`.
    :param target_mu_x_threshold: Distance from target mu_x below which mu_x is
        considered to be achieved.
    :param target_mu_y_threshold: Distance from target mu_y below which mu_y is
        considered to be achieved.
    :param target_sigma_x_threshold: Distance from target sigma_x below which sigma_x is
        considered to be achieved.
    :param target_sigma_y_threshold: Distance from target sigma_y below which sigma_y is
        considered to be achieved.
    :param threshold_hold: Number of steps that all beam parameters difference must be
        below their thresolds before an episode is terminated as successful.
    :param w_beam: Weight of all beam parameter-related rewards in the total reward.
    :param w_done: Weight of the successful episode termination bonus in the total
        reward.
    :param w_mu_x: Weight of the mu_x component in the beam parameter-related reward.
    :param w_mu_x_in_threshold: Weight of the bonus reward when mu_x is within its
        threshold from the target.
    :param w_mu_y: Weight of the mu_y component in the beam parameter-related reward.
    :param w_mu_y_in_threshold: Weight of the bonus reward when mu_y is within its
        threshold from the target.
    :param w_sigma_x: Weight of the sigma_x component in the beam parameter-related
        reward.
    :param w_sigma_x_in_threshold: Weight of the bonus reward when sigma_x is within its
        threshold from the target.
    :param w_sigma_y: Weight of the sigma_y component in the beam parameter-related
        reward.
    :param w_sigma_y_in_threshold: Weight of the bonus reward when sigma_y is within its
        threshold from the target.
    :param w_time: Weight of the reward received for each passing time step.
    """

    def __init__(
        self,
        backend: TransverseTuningBaseBackend,
        action_mode: str = "direct",
        beam_distance_ord: int = 1,
        logarithmic_beam_distance: bool = False,
        magnet_init_mode: Optional[str] = None,
        magnet_init_values: Optional[np.ndarray] = None,
        max_quad_delta: Optional[float] = None,
        max_steerer_delta: Optional[float] = None,
        normalize_beam_distance: bool = True,
        reward_mode: str = "differential",
        target_beam_mode: str = "random",
        target_beam_values: Optional[np.ndarray] = None,
        target_mu_x_threshold: float = 3.3198e-6,
        target_mu_y_threshold: float = 2.4469e-6,
        target_sigma_x_threshold: float = 3.3198e-6,
        target_sigma_y_threshold: float = 2.4469e-6,
        threshold_hold: int = 1,
        unidirectional_quads: bool = False,
        w_beam: float = 0.0,
        w_done: float = 0.0,
        w_mu_x: float = 0.0,
        w_mu_x_in_threshold: float = 0.0,
        w_mu_y: float = 0.0,
        w_mu_y_in_threshold: float = 0.0,
        w_on_screen: float = 0.0,
        w_sigma_x: float = 0.0,
        w_sigma_x_in_threshold: float = 0.0,
        w_sigma_y: float = 0.0,
        w_sigma_y_in_threshold: float = 0.0,
        w_time: float = 0.0,
    ) -> None:
        self.backend = backend

        self.action_mode = action_mode
        self.beam_distance_ord = beam_distance_ord
        self.logarithmic_beam_distance = logarithmic_beam_distance
        self.magnet_init_mode = magnet_init_mode
        self.magnet_init_values = magnet_init_values
        self.max_quad_delta = max_quad_delta
        self.max_steerer_delta = max_steerer_delta
        self.normalize_beam_distance = normalize_beam_distance
        self.reward_mode = reward_mode
        self.target_beam_mode = target_beam_mode
        self.target_beam_values = target_beam_values
        self.target_mu_x_threshold = target_mu_x_threshold
        self.target_mu_y_threshold = target_mu_y_threshold
        self.target_sigma_x_threshold = target_sigma_x_threshold
        self.target_sigma_y_threshold = target_sigma_y_threshold
        self.threshold_hold = threshold_hold
        self.unidirectional_quads = unidirectional_quads
        self.w_beam = w_beam
        self.w_done = w_done
        self.w_mu_x = w_mu_x
        self.w_mu_x_in_threshold = w_mu_x_in_threshold
        self.w_mu_y = w_mu_y
        self.w_mu_y_in_threshold = w_mu_y_in_threshold
        self.w_on_screen = w_on_screen
        self.w_sigma_x = w_sigma_x
        self.w_sigma_x_in_threshold = w_sigma_x_in_threshold
        self.w_sigma_y = w_sigma_y
        self.w_sigma_y_in_threshold = w_sigma_y_in_threshold
        self.w_time = w_time

        if unidirectional_quads:
            self.magnet_space = spaces.Box(
                low=np.array([0, -72, -6.1782e-3, -6.1782e-3, 0], dtype=np.float32),
                high=np.array(
                    [72, 0, 6.1782e-3, 6.1782e-3, 72],
                    dtype=np.float32,
                ),
            )
        else:
            self.magnet_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -6.1782e-3, -72], dtype=np.float32),
                high=np.array(
                    [72, 72, 6.1782e-3, 6.1782e-3, 72],
                    dtype=np.float32,
                ),
            )

        # Create action space
        if self.action_mode == "direct":
            self.action_space = self.magnet_space
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=np.array(
                    [
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                        -self.max_steerer_delta,
                        -self.max_steerer_delta,
                        -self.max_quad_delta,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        self.max_quad_delta,
                        self.max_quad_delta,
                        self.max_steerer_delta,
                        self.max_steerer_delta,
                        self.max_quad_delta,
                    ],
                    dtype=np.float32,
                ),
            )
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

        # Create observation space
        self.observation_space = spaces.Dict(
            {
                "beam": spaces.Box(
                    low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                ),
                "magnets": self.magnet_space,
                "target": spaces.Box(
                    low=np.array([-2e-3, 0, -2e-3, 0], dtype=np.float32),
                    high=np.array([2e-3, 2e-3, 2e-3, 2e-3], dtype=np.float32),
                ),
            }
        )

        # Setup the accelerator (either simulation or the actual machine)
        self.backend.setup()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        # TODO Update (and probably refactor) this render function
        raise RuntimeError(
            f"{self.__class__.__name__} is using the render() function from"
            " EATransverseTuning that is borken. Don't use!"
        )

        assert mode == "rgb_array" or mode == "human"

        binning = self.backend.get_binning()
        pixel_size = self.backend.get_pixel_size()
        resolution = self.backend.get_screen_resolution()

        # Read screen image and make 8-bit RGB
        img = self.backend.get_screen_image()
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)

        # Redraw beam image as if it were binning = 4
        render_resolution = (resolution * binning / 4).astype("int")
        img = cv2.resize(img, render_resolution)

        # Draw desired ellipse
        tb = self.target_beam
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(tb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(tb[1] / pixel_size_b4[0])
        e_pos_y = int(-tb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(tb[3] / pixel_size_b4[1])
        blue = (255, 204, 79)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, blue, 2
        )

        # Draw beam ellipse
        cb = self.backend.get_beam_parameters()
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(cb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(cb[1] / pixel_size_b4[0])
        e_pos_y = int(-cb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(cb[3] / pixel_size_b4[1])
        red = (0, 0, 255)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, red, 2
        )

        # Adjust aspect ratio
        new_width = int(img.shape[1] * pixel_size_b4[0] / pixel_size_b4[1])
        img = cv2.resize(img, (new_width, img.shape[0]))

        # Add magnet values and beam parameters
        magnets = self.backend.get_magnets()
        padding = np.full(
            (int(img.shape[0] * 0.27), img.shape[1], 3), fill_value=255, dtype=np.uint8
        )
        img = np.vstack([img, padding])
        black = (0, 0, 0)
        red = (0, 0, 255)
        green = (0, 255, 0)
        img = cv2.putText(
            img, f"Q1={magnets[0]:.2f}", (15, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img, f"Q2={magnets[1]:.2f}", (215, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CV={magnets[2]*1e3:.2f}",
            (415, 545),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        img = cv2.putText(
            img, f"Q3={magnets[3]:.2f}", (615, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CH={magnets[4]*1e3:.2f}",
            (15, 585),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        mu_x_color = black
        if self.target_mu_x_threshold != np.inf:
            mu_x_color = (
                green if abs(cb[0] - tb[0]) < self.target_mu_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"mx={cb[0]*1e3:.2f}",
            (15, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_x_color,
        )
        sigma_x_color = black
        if self.target_sigma_x_threshold != np.inf:
            sigma_x_color = (
                green if abs(cb[1] - tb[1]) < self.target_sigma_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"sx={cb[1]*1e3:.2f}",
            (215, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_x_color,
        )
        mu_y_color = black
        if self.target_mu_y_threshold != np.inf:
            mu_y_color = (
                green if abs(cb[2] - tb[2]) < self.target_mu_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"my={cb[2]*1e3:.2f}",
            (415, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_y_color,
        )
        sigma_y_color = black
        if self.target_sigma_y_threshold != np.inf:
            sigma_y_color = (
                green if abs(cb[3] - tb[3]) < self.target_sigma_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"sy={cb[3]*1e3:.2f}",
            (615, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_y_color,
        )

        if mode == "human":
            cv2.imshow("ARES EA", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class DLTransverseTuning(TransverseTuningEnv):
    """
    Environment for positioning and focusing the beam on ARDLBSCR1 using ARDLMCVM1,
    ARDLCHM1, ARDLMQZM1 and ARDLMQZM2.

    :param backend: Backend for communication with either a simulation or the control
        system
    :param action_mode: Choose weather actions set magnet settings directly (`"direct"`)
        or change magnet settings (`"delta"`).
    :param beam_distance_ord: Order of distance to use to compute distance between
        current beam and target beam.
    :param logarithmic_beam_distance: Whether to take the logarithm of the beam
        distance.
    :param magnet_init_mode: Magnet initialisation on `reset`. Set to `None` for magnets
        to stay at their current settings, `"random"` to be set to random settings or
        `"constant"` to set them to the settings given by `magnet_init_values`.
    :param magnet_init_values: Values to set magnets to on `reset`. Is only used when
        `magnet_init_mode` is set to `"constant"`.
    :param max_quad_delta: Limit of by how much quadrupole settings may be changed when
        `action_mode` is set to `"delta"`.
    :param max_steerer_delta: Limit of by how much steerer settings may be changed when
        `action_mode` is set to `"delta"`.
    :param normalize_beam_distance: Whether to divide beam distance by the initial beam
        distance in order to scale it to a value around 1.
    :param reward_mode: Set to `"feedback"` to compute the reward as the negative beam
        distance. Set to `"differential"` to compute the reward as the reduction of the
        beam distance.
    :param target_beam_mode: Setting of target beam on `reset`. Choose from `"constant"`
        or `"random"`. The `"constant"` setting requires `target_beam_values` to be set.
    :param target_beam_values: Target beam to use when `target_beam_mode` is set to
        `"constant"`.
    :param target_mu_x_threshold: Distance from target mu_x below which mu_x is
        considered to be achieved.
    :param target_mu_y_threshold: Distance from target mu_y below which mu_y is
        considered to be achieved.
    :param target_sigma_x_threshold: Distance from target sigma_x below which sigma_x is
        considered to be achieved.
    :param target_sigma_y_threshold: Distance from target sigma_y below which sigma_y is
        considered to be achieved.
    :param threshold_hold: Number of steps that all beam parameters difference must be
        below their thresolds before an episode is terminated as successful.
    :param w_beam: Weight of all beam parameter-related rewards in the total reward.
    :param w_done: Weight of the successful episode termination bonus in the total
        reward.
    :param w_mu_x: Weight of the mu_x component in the beam parameter-related reward.
    :param w_mu_x_in_threshold: Weight of the bonus reward when mu_x is within its
        threshold from the target.
    :param w_mu_y: Weight of the mu_y component in the beam parameter-related reward.
    :param w_mu_y_in_threshold: Weight of the bonus reward when mu_y is within its
        threshold from the target.
    :param w_sigma_x: Weight of the sigma_x component in the beam parameter-related
        reward.
    :param w_sigma_x_in_threshold: Weight of the bonus reward when sigma_x is within its
        threshold from the target.
    :param w_sigma_y: Weight of the sigma_y component in the beam parameter-related
        reward.
    :param w_sigma_y_in_threshold: Weight of the bonus reward when sigma_y is within its
        threshold from the target.
    :param w_time: Weight of the reward received for each passing time step.
    """

    def __init__(
        self,
        backend: TransverseTuningBaseBackend,
        action_mode: str = "direct",
        beam_distance_ord: int = 1,
        logarithmic_beam_distance: bool = False,
        magnet_init_mode: Optional[str] = None,
        magnet_init_values: Optional[np.ndarray] = None,
        max_quad_delta: Optional[float] = None,
        max_steerer_delta: Optional[float] = None,
        normalize_beam_distance: bool = True,
        reward_mode: str = "differential",
        target_beam_mode: str = "random",
        target_beam_values: Optional[np.ndarray] = None,
        target_mu_x_threshold: float = 3.3198e-6,
        target_mu_y_threshold: float = 2.4469e-6,
        target_sigma_x_threshold: float = 3.3198e-6,
        target_sigma_y_threshold: float = 2.4469e-6,
        threshold_hold: int = 1,
        unidirectional_quads: bool = False,
        w_beam: float = 0.0,
        w_done: float = 0.0,
        w_mu_x: float = 0.0,
        w_mu_x_in_threshold: float = 0.0,
        w_mu_y: float = 0.0,
        w_mu_y_in_threshold: float = 0.0,
        w_on_screen: float = 0.0,
        w_sigma_x: float = 0.0,
        w_sigma_x_in_threshold: float = 0.0,
        w_sigma_y: float = 0.0,
        w_sigma_y_in_threshold: float = 0.0,
        w_time: float = 0.0,
    ) -> None:
        self.backend = backend

        self.action_mode = action_mode
        self.beam_distance_ord = beam_distance_ord
        self.logarithmic_beam_distance = logarithmic_beam_distance
        self.magnet_init_mode = magnet_init_mode
        self.magnet_init_values = magnet_init_values
        self.max_quad_delta = max_quad_delta
        self.max_steerer_delta = max_steerer_delta
        self.normalize_beam_distance = normalize_beam_distance
        self.reward_mode = reward_mode
        self.target_beam_mode = target_beam_mode
        self.target_beam_values = target_beam_values
        self.target_mu_x_threshold = target_mu_x_threshold
        self.target_mu_y_threshold = target_mu_y_threshold
        self.target_sigma_x_threshold = target_sigma_x_threshold
        self.target_sigma_y_threshold = target_sigma_y_threshold
        self.threshold_hold = threshold_hold
        self.unidirectional_quads = unidirectional_quads
        self.w_beam = w_beam
        self.w_done = w_done
        self.w_mu_x = w_mu_x
        self.w_mu_x_in_threshold = w_mu_x_in_threshold
        self.w_mu_y = w_mu_y
        self.w_mu_y_in_threshold = w_mu_y_in_threshold
        self.w_on_screen = w_on_screen
        self.w_sigma_x = w_sigma_x
        self.w_sigma_x_in_threshold = w_sigma_x_in_threshold
        self.w_sigma_y = w_sigma_y
        self.w_sigma_y_in_threshold = w_sigma_y_in_threshold
        self.w_time = w_time

        if unidirectional_quads:
            self.magnet_space = spaces.Box(
                low=np.array([-6.1782e-3, -6.1782e-3, 0, -72], dtype=np.float32),
                high=np.array([6.1782e-3, 6.1782e-3, 72, 0], dtype=np.float32),
            )
        else:
            self.magnet_space = spaces.Box(
                low=np.array([-6.1782e-3, -6.1782e-3, -72, -72], dtype=np.float32),
                high=np.array([6.1782e-3, 6.1782e-3, 72, 72], dtype=np.float32),
            )

        # Create action space
        if self.action_mode == "direct":
            self.action_space = self.magnet_space
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=np.array(
                    [
                        -self.max_steerer_delta,
                        -self.max_steerer_delta,
                        -self.max_quad_delta,
                        -self.max_quad_delta,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        self.max_steerer_delta,
                        self.max_steerer_delta,
                        self.max_quad_delta,
                        self.max_quad_delta,
                    ],
                    dtype=np.float32,
                ),
            )
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

        # Create observation space
        self.observation_space = spaces.Dict(
            {
                "beam": spaces.Box(
                    low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                ),
                "magnets": self.magnet_space,
                "target": spaces.Box(
                    low=np.array([-2e-3, 0, -2e-3, 0], dtype=np.float32),
                    high=np.array([2e-3, 2e-3, 2e-3, 2e-3], dtype=np.float32),
                ),
            }
        )

        # Setup the accelerator (either simulation or the actual machine)
        self.backend.setup()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        # TODO Update (and probably refactor) this render function
        raise RuntimeError(
            f"{self.__class__.__name__} is using the render() function from"
            " EATransverseTuning that is borken. Don't use!"
        )

        assert mode == "rgb_array" or mode == "human"

        binning = self.backend.get_binning()
        pixel_size = self.backend.get_pixel_size()
        resolution = self.backend.get_screen_resolution()

        # Read screen image and make 8-bit RGB
        img = self.backend.get_screen_image()
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)

        # Redraw beam image as if it were binning = 4
        render_resolution = (resolution * binning / 4).astype("int")
        img = cv2.resize(img, render_resolution)

        # Draw desired ellipse
        tb = self.target_beam
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(tb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(tb[1] / pixel_size_b4[0])
        e_pos_y = int(-tb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(tb[3] / pixel_size_b4[1])
        blue = (255, 204, 79)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, blue, 2
        )

        # Draw beam ellipse
        cb = self.backend.get_beam_parameters()
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(cb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(cb[1] / pixel_size_b4[0])
        e_pos_y = int(-cb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(cb[3] / pixel_size_b4[1])
        red = (0, 0, 255)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, red, 2
        )

        # Adjust aspect ratio
        new_width = int(img.shape[1] * pixel_size_b4[0] / pixel_size_b4[1])
        img = cv2.resize(img, (new_width, img.shape[0]))

        # Add magnet values and beam parameters
        magnets = self.backend.get_magnets()
        padding = np.full(
            (int(img.shape[0] * 0.27), img.shape[1], 3), fill_value=255, dtype=np.uint8
        )
        img = np.vstack([img, padding])
        black = (0, 0, 0)
        red = (0, 0, 255)
        green = (0, 255, 0)
        img = cv2.putText(
            img, f"Q1={magnets[0]:.2f}", (15, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img, f"Q2={magnets[1]:.2f}", (215, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CV={magnets[2]*1e3:.2f}",
            (415, 545),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        img = cv2.putText(
            img, f"Q3={magnets[3]:.2f}", (615, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CH={magnets[4]*1e3:.2f}",
            (15, 585),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        mu_x_color = black
        if self.target_mu_x_threshold != np.inf:
            mu_x_color = (
                green if abs(cb[0] - tb[0]) < self.target_mu_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"mx={cb[0]*1e3:.2f}",
            (15, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_x_color,
        )
        sigma_x_color = black
        if self.target_sigma_x_threshold != np.inf:
            sigma_x_color = (
                green if abs(cb[1] - tb[1]) < self.target_sigma_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"sx={cb[1]*1e3:.2f}",
            (215, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_x_color,
        )
        mu_y_color = black
        if self.target_mu_y_threshold != np.inf:
            mu_y_color = (
                green if abs(cb[2] - tb[2]) < self.target_mu_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"my={cb[2]*1e3:.2f}",
            (415, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_y_color,
        )
        sigma_y_color = black
        if self.target_sigma_y_threshold != np.inf:
            sigma_y_color = (
                green if abs(cb[3] - tb[3]) < self.target_sigma_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"sy={cb[3]*1e3:.2f}",
            (615, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_y_color,
        )

        if mode == "human":
            cv2.imshow("ARES EA", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class SHTransverseTuning(TransverseTuningEnv):
    """
    Environment for positioning and focusing the beam on ARSHBSCE2 using ARDLMCVM2,
    ARDLMQZM3, ARDLMCHM2 and ARDLMQZM4.

    :param backend: Backend for communication with either a simulation or the control
        system
    :param action_mode: Choose weather actions set magnet settings directly (`"direct"`)
        or change magnet settings (`"delta"`).
    :param beam_distance_ord: Order of distance to use to compute distance between
        current beam and target beam.
    :param logarithmic_beam_distance: Whether to take the logarithm of the beam
        distance.
    :param magnet_init_mode: Magnet initialisation on `reset`. Set to `None` for magnets
        to stay at their current settings, `"random"` to be set to random settings or
        `"constant"` to set them to the settings given by `magnet_init_values`.
    :param magnet_init_values: Values to set magnets to on `reset`. Is only used when
        `magnet_init_mode` is set to `"constant"`.
    :param max_quad_delta: Limit of by how much quadrupole settings may be changed when
        `action_mode` is set to `"delta"`.
    :param max_steerer_delta: Limit of by how much steerer settings may be changed when
        `action_mode` is set to `"delta"`.
    :param normalize_beam_distance: Whether to divide beam distance by the initial beam
        distance in order to scale it to a value around 1.
    :param reward_mode: Set to `"feedback"` to compute the reward as the negative beam
        distance. Set to `"differential"` to compute the reward as the reduction of the
        beam distance.
    :param target_beam_mode: Setting of target beam on `reset`. Choose from `"constant"`
        or `"random"`. The `"constant"` setting requires `target_beam_values` to be set.
    :param target_beam_values: Target beam to use when `target_beam_mode` is set to
        `"constant"`.
    :param target_mu_x_threshold: Distance from target mu_x below which mu_x is
        considered to be achieved.
    :param target_mu_y_threshold: Distance from target mu_y below which mu_y is
        considered to be achieved.
    :param target_sigma_x_threshold: Distance from target sigma_x below which sigma_x is
        considered to be achieved.
    :param target_sigma_y_threshold: Distance from target sigma_y below which sigma_y is
        considered to be achieved.
    :param threshold_hold: Number of steps that all beam parameters difference must be
        below their thresolds before an episode is terminated as successful.
    :param w_beam: Weight of all beam parameter-related rewards in the total reward.
    :param w_done: Weight of the successful episode termination bonus in the total
        reward.
    :param w_mu_x: Weight of the mu_x component in the beam parameter-related reward.
    :param w_mu_x_in_threshold: Weight of the bonus reward when mu_x is within its
        threshold from the target.
    :param w_mu_y: Weight of the mu_y component in the beam parameter-related reward.
    :param w_mu_y_in_threshold: Weight of the bonus reward when mu_y is within its
        threshold from the target.
    :param w_sigma_x: Weight of the sigma_x component in the beam parameter-related
        reward.
    :param w_sigma_x_in_threshold: Weight of the bonus reward when sigma_x is within its
        threshold from the target.
    :param w_sigma_y: Weight of the sigma_y component in the beam parameter-related
        reward.
    :param w_sigma_y_in_threshold: Weight of the bonus reward when sigma_y is within its
        threshold from the target.
    :param w_time: Weight of the reward received for each passing time step.
    """

    def __init__(
        self,
        backend: TransverseTuningBaseBackend,
        action_mode: str = "direct",
        beam_distance_ord: int = 1,
        logarithmic_beam_distance: bool = False,
        magnet_init_mode: Optional[str] = None,
        magnet_init_values: Optional[np.ndarray] = None,
        max_quad_delta: Optional[float] = None,
        max_steerer_delta: Optional[float] = None,
        normalize_beam_distance: bool = True,
        reward_mode: str = "differential",
        target_beam_mode: str = "random",
        target_beam_values: Optional[np.ndarray] = None,
        target_mu_x_threshold: float = 3.3198e-6,
        target_mu_y_threshold: float = 2.4469e-6,
        target_sigma_x_threshold: float = 3.3198e-6,
        target_sigma_y_threshold: float = 2.4469e-6,
        threshold_hold: int = 1,
        unidirectional_quads: bool = False,
        w_beam: float = 0.0,
        w_done: float = 0.0,
        w_mu_x: float = 0.0,
        w_mu_x_in_threshold: float = 0.0,
        w_mu_y: float = 0.0,
        w_mu_y_in_threshold: float = 0.0,
        w_on_screen: float = 0.0,
        w_sigma_x: float = 0.0,
        w_sigma_x_in_threshold: float = 0.0,
        w_sigma_y: float = 0.0,
        w_sigma_y_in_threshold: float = 0.0,
        w_time: float = 0.0,
    ) -> None:
        self.backend = backend

        self.action_mode = action_mode
        self.beam_distance_ord = beam_distance_ord
        self.logarithmic_beam_distance = logarithmic_beam_distance
        self.magnet_init_mode = magnet_init_mode
        self.magnet_init_values = magnet_init_values
        self.max_quad_delta = max_quad_delta
        self.max_steerer_delta = max_steerer_delta
        self.normalize_beam_distance = normalize_beam_distance
        self.reward_mode = reward_mode
        self.target_beam_mode = target_beam_mode
        self.target_beam_values = target_beam_values
        self.target_mu_x_threshold = target_mu_x_threshold
        self.target_mu_y_threshold = target_mu_y_threshold
        self.target_sigma_x_threshold = target_sigma_x_threshold
        self.target_sigma_y_threshold = target_sigma_y_threshold
        self.threshold_hold = threshold_hold
        self.unidirectional_quads = unidirectional_quads
        self.w_beam = w_beam
        self.w_done = w_done
        self.w_mu_x = w_mu_x
        self.w_mu_x_in_threshold = w_mu_x_in_threshold
        self.w_mu_y = w_mu_y
        self.w_mu_y_in_threshold = w_mu_y_in_threshold
        self.w_on_screen = w_on_screen
        self.w_sigma_x = w_sigma_x
        self.w_sigma_x_in_threshold = w_sigma_x_in_threshold
        self.w_sigma_y = w_sigma_y
        self.w_sigma_y_in_threshold = w_sigma_y_in_threshold
        self.w_time = w_time

        if unidirectional_quads:
            self.magnet_space = spaces.Box(
                low=np.array([-6.1782e-3, 0, -6.1782e-3, -72], dtype=np.float32),
                high=np.array([6.1782e-3, 72, 6.1782e-3, 0], dtype=np.float32),
            )
        else:
            self.magnet_space = spaces.Box(
                low=np.array([-6.1782e-3, -72, -6.1782e-3, -72], dtype=np.float32),
                high=np.array([6.1782e-3, 72, 6.1782e-3, 72], dtype=np.float32),
            )

        # Create action space
        if self.action_mode == "direct":
            self.action_space = self.magnet_space
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=np.array(
                    [
                        -self.max_steerer_delta,
                        -self.max_quad_delta,
                        -self.max_steerer_delta,
                        -self.max_quad_delta,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        self.max_steerer_delta,
                        self.max_quad_delta,
                        self.max_steerer_delta,
                        self.max_quad_delta,
                    ],
                    dtype=np.float32,
                ),
            )
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

        # Create observation space
        self.observation_space = spaces.Dict(
            {
                "beam": spaces.Box(
                    low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                ),
                "magnets": self.magnet_space,
                "target": spaces.Box(
                    low=np.array([-2e-3, 0, -2e-3, 0], dtype=np.float32),
                    high=np.array([2e-3, 2e-3, 2e-3, 2e-3], dtype=np.float32),
                ),
            }
        )

        # Setup the accelerator (either simulation or the actual machine)
        self.backend.setup()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        # TODO Update (and probably refactor) this render function
        raise RuntimeError(
            f"{self.__class__.__name__} is using the render() function from"
            " EATransverseTuning that is borken. Don't use!"
        )

        assert mode == "rgb_array" or mode == "human"

        binning = self.backend.get_binning()
        pixel_size = self.backend.get_pixel_size()
        resolution = self.backend.get_screen_resolution()

        # Read screen image and make 8-bit RGB
        img = self.backend.get_screen_image()
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)

        # Redraw beam image as if it were binning = 4
        render_resolution = (resolution * binning / 4).astype("int")
        img = cv2.resize(img, render_resolution)

        # Draw desired ellipse
        tb = self.target_beam
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(tb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(tb[1] / pixel_size_b4[0])
        e_pos_y = int(-tb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(tb[3] / pixel_size_b4[1])
        blue = (255, 204, 79)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, blue, 2
        )

        # Draw beam ellipse
        cb = self.backend.get_beam_parameters()
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(cb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(cb[1] / pixel_size_b4[0])
        e_pos_y = int(-cb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(cb[3] / pixel_size_b4[1])
        red = (0, 0, 255)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, red, 2
        )

        # Adjust aspect ratio
        new_width = int(img.shape[1] * pixel_size_b4[0] / pixel_size_b4[1])
        img = cv2.resize(img, (new_width, img.shape[0]))

        # Add magnet values and beam parameters
        magnets = self.backend.get_magnets()
        padding = np.full(
            (int(img.shape[0] * 0.27), img.shape[1], 3), fill_value=255, dtype=np.uint8
        )
        img = np.vstack([img, padding])
        black = (0, 0, 0)
        red = (0, 0, 255)
        green = (0, 255, 0)
        img = cv2.putText(
            img, f"Q1={magnets[0]:.2f}", (15, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img, f"Q2={magnets[1]:.2f}", (215, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CV={magnets[2]*1e3:.2f}",
            (415, 545),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        img = cv2.putText(
            img, f"Q3={magnets[3]:.2f}", (615, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CH={magnets[4]*1e3:.2f}",
            (15, 585),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        mu_x_color = black
        if self.target_mu_x_threshold != np.inf:
            mu_x_color = (
                green if abs(cb[0] - tb[0]) < self.target_mu_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"mx={cb[0]*1e3:.2f}",
            (15, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_x_color,
        )
        sigma_x_color = black
        if self.target_sigma_x_threshold != np.inf:
            sigma_x_color = (
                green if abs(cb[1] - tb[1]) < self.target_sigma_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"sx={cb[1]*1e3:.2f}",
            (215, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_x_color,
        )
        mu_y_color = black
        if self.target_mu_y_threshold != np.inf:
            mu_y_color = (
                green if abs(cb[2] - tb[2]) < self.target_mu_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"my={cb[2]*1e3:.2f}",
            (415, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_y_color,
        )
        sigma_y_color = black
        if self.target_sigma_y_threshold != np.inf:
            sigma_y_color = (
                green if abs(cb[3] - tb[3]) < self.target_sigma_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"sy={cb[3]*1e3:.2f}",
            (615, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_y_color,
        )

        if mode == "human":
            cv2.imshow("ARES EA", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
