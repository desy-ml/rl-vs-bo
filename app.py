from datetime import datetime
import pickle
import sys
from threading import Event
from time import sleep

from gym.wrappers import FlattenObservation, Monitor, TimeLimit
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import pydoocs
from stable_baselines3 import TD3

from environments.machine import ARESEAMachine


class LiveScreenView(FigureCanvasQTAgg):

    def __init__(self, resolution, pixel_size):
        self.resolution = resolution
        self.pixel_size = pixel_size

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        
        super().__init__(self.fig)

        self.screen_extent = (-self.resolution[0] * self.pixel_size[0] / 2 * 1e3,
                              self.resolution[0] * self.pixel_size[0] / 2 * 1e3,
                              -self.resolution[1] * self.pixel_size[1] / 2 * 1e3,
                              self.resolution[1] * self.pixel_size[1] / 2 * 1e3)
        self.create_plot()

        self.setFixedSize(600, 420)
    
    def create_plot(self):
        self.screen_plot = self.ax.imshow(
            np.zeros(self.resolution), 
            cmap="magma",
            interpolation="None",
            extent=self.screen_extent
        )
        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("y (mm)")
        self.ax.set_title("Live View AR.EA.BSC.R.1")

        self.mu_x, self.mu_y, self.sigma_x, self.sigma_y = [0] * 4
        self.select_ellipse = Ellipse(
            (self.mu_x,self.mu_y),
            self.sigma_x,
            self.sigma_y,
            fill=False,
            color="white"
        )
        self.ax.add_patch(self.select_ellipse)

        self.fig.tight_layout()
    
    @qtc.pyqtSlot(np.ndarray)
    def update_screen_data(self, screen_data):
        self.screen_plot.set_data(screen_data)
        self.screen_plot.set_clim(vmin=0, vmax=screen_data.max())

        self.draw()
    
    def update_target(self, mu_x, mu_y, sigma_x, sigma_y):
        mu_x, mu_y, sigma_x, sigma_y = [x * 1e3 for x in (mu_x, mu_y, sigma_x, sigma_y)]

        self.select_ellipse.set_center((mu_x, mu_y))
        self.select_ellipse.set_width(2 * sigma_x)
        self.select_ellipse.set_height(2 * sigma_y)

        self.draw()


class AgentScreenView(FigureCanvasQTAgg):

    def __init__(self, resolution, pixel_size):
        self.resolution = resolution
        self.pixel_size = pixel_size

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        
        super().__init__(self.fig)

        self.screen_extent = (-self.resolution[0] * self.pixel_size[0] / 2 * 1e3,
                              self.resolution[0] * self.pixel_size[0] / 2 * 1e3,
                              -self.resolution[1] * self.pixel_size[1] / 2 * 1e3,
                              self.resolution[1] * self.pixel_size[1] / 2 * 1e3)
        self.create_plot()

        self.setFixedSize(600, 420)
    
    def create_plot(self):
        self.screen_plot = self.ax.imshow(
            np.zeros(self.resolution), 
            cmap="magma",
            interpolation="None",
            extent=self.screen_extent
        )
        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("y (mm)")
        self.ax.set_title("Agent View")

        self.achieved_mu_x, self.achieved_mu_y, self.achieved_sigma_x, self.achieved_sigma_y = [0] * 4
        self.achieved_ellipse = Ellipse(
            (self.achieved_mu_x,self.achieved_mu_y),
            self.achieved_sigma_x,
            self.achieved_sigma_y,
            fill=False,
            color="deepskyblue",
            linestyle="--"
        )
        self.ax.add_patch(self.achieved_ellipse)

        self.desired_mu_x, self.desired_mu_y, self.desired_sigma_x, self.desired_sigma_y = [0] * 4
        self.desired_ellipse = Ellipse(
            (self.desired_mu_x,self.desired_mu_y),
            self.desired_sigma_x,
            self.desired_sigma_y,
            fill=False,
            color="white"
        )
        self.ax.add_patch(self.desired_ellipse)

        self.fig.tight_layout()
    
    @qtc.pyqtSlot(np.ndarray)
    def update_screen_data(self, screen_data):
        self.screen_plot.set_data(screen_data)
        self.screen_plot.set_clim(vmin=0, vmax=screen_data.max())

        self.draw()
    
    @qtc.pyqtSlot(float, float, float, float)
    def update_achieved(self, mu_x, mu_y, sigma_x, sigma_y):
        mu_x, mu_y, sigma_x, sigma_y = [x * 1e3 for x in (mu_x, mu_y, sigma_x, sigma_y)]

        self.achieved_ellipse.set_center((mu_x, mu_y))
        self.achieved_ellipse.set_width(2 * sigma_x)
        self.achieved_ellipse.set_height(2 * sigma_y)

        self.draw()
    
    @qtc.pyqtSlot(float, float, float, float)
    def update_desired(self, mu_x, mu_y, sigma_x, sigma_y):
        mu_x, mu_y, sigma_x, sigma_y = [x * 1e3 for x in (mu_x, mu_y, sigma_x, sigma_y)]

        self.desired_ellipse.set_center((mu_x, mu_y))
        self.desired_ellipse.set_width(2 * sigma_x)
        self.desired_ellipse.set_height(2 * sigma_y)
        
        self.draw()


class AcceleratorReadThread(qtc.QThread):

    screen_updated = qtc.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()

    def run(self):
        while True:
            self.read_screen()
            sleep(0.1)
    
    def read_screen(self):
        response = pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ")
        screen_data = response["data"]
        
        self.screen_updated.emit(screen_data)


class AgentThread(qtc.QThread):
    
    done = qtc.pyqtSignal(tuple)
    agent_screen_updated = qtc.pyqtSignal(np.ndarray)
    want_step_permission = qtc.pyqtSignal(np.ndarray, np.ndarray)
    took_step = qtc.pyqtSignal(int)
    desired_updated = qtc.pyqtSignal(float, float, float, float)
    achieved_updated = qtc.pyqtSignal(float, float, float, float)
    done = qtc.pyqtSignal(int, np.ndarray)

    step_permission_event = Event()

    def __init__(self, model_name, desired_goal, target_delta):
        super().__init__()

        self.model_name = model_name
        self.desired_goal = desired_goal
        self.target_delta = target_delta

        self.step_permission_event.clear()
        self.step_permission = False

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    def run(self):
        self.took_step.emit(0)
        self.desired_updated.emit(*self.desired_goal)

        self.env = ARESEAMachine()
        self.env = TimeLimit(self.env, max_episode_steps=50)
        self.env = FlattenObservation(self.env)
        self.env = Monitor(self.env,
                           f"experiments/{self.timestamp}/recording",
                           video_callable=lambda i: True)
        
        self.env.unwrapped.target_delta = self.target_delta

        model = TD3.load(f"models/{self.model_name}")

        done = False
        i = 0
        observation = self.env.reset(goal=self.desired_goal)
        log = {
            "backgrounds": [self.env.unwrapped.backgrounds],
            "background": [self.env.unwrapped.background],
            "beams": [self.env.unwrapped.beams],
            "beam": [self.env.unwrapped.beam],
            "screen_data": [self.env.unwrapped.screen_data],
            "observation": [self.env.unwrapped.observation],
            "action": []
        }
        self.agent_screen_updated.emit(self.env.screen_data)
        self.achieved_updated.emit(*self.env.unwrapped.observation["achieved_goal"])
        while not done:
            action, _ = model.predict(observation)

            if not self.ask_step_permission(action, observation):
                print("Permission denied!")
                break
            print("Permission granted!")

            observation, _, done, _ = self.env.step(action)

            log["backgrounds"].append(self.env.unwrapped.backgrounds)
            log["background"].append(self.env.unwrapped.background)
            log["beams"].append(self.env.unwrapped.beams)
            log["beam"].append(self.env.unwrapped.beam)
            log["screen_data"].append(self.env.unwrapped.screen_data)
            log["observation"].append(self.env.unwrapped.observation)
            log["action"].append(self.env.unwrapped.action2accelerator(action))

            self.agent_screen_updated.emit(self.env.unwrapped.screen_data)
            self.achieved_updated.emit(*self.env.unwrapped.observation["achieved_goal"])
            i += 1
            self.took_step.emit(i)
        
        self.took_step.emit(50)

        self.env.close()

        log["history"] = self.env.unwrapped.history
        log["model_name"] = self.model_name
        log["target_delta"] = self.env.unwrapped.target_delta
        logpath = f"experiments/{self.timestamp}/log.pkl"
        with open(logpath, "wb") as f:
            pickle.dump(log, f)
            print(f"Log file saved as \"{logpath}\"")
        
        desired = self.env.unwrapped.observation["desired_goal"]
        achieved = self.env.unwrapped.observation["achieved_goal"]
        delta = np.abs(desired - achieved)
        self.done.emit(i, delta)

    def ask_step_permission(self, action, observation):
        old_actuators = observation[-5:] * self.env.accelerator_observation_space["observation"].high[-5:]
        new_actuators = old_actuators + action * self.env.accelerator_action_space.high

        self.want_step_permission.emit(old_actuators, new_actuators)
        self.step_permission_event.wait()
        self.step_permission_event.clear()

        tmp = self.step_permission
        self.step_permission = False

        return tmp


class HorizontalSeperationLine(qtw.QFrame):

  def __init__(self):
    super().__init__()
    self.setMinimumWidth(1)
    self.setFixedHeight(20)
    self.setFrameShape(qtw.QFrame.HLine)
    self.setFrameShadow(qtw.QFrame.Sunken)
    self.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Minimum)


class App(qtw.QWidget):

    achieved_updated = qtc.pyqtSignal(float, float, float, float)
    desired_updated = qtc.pyqtSignal(float, float, float, float)
    deltas_updated = qtc.pyqtSignal(float, float, float, float)
    magnet_read = qtc.pyqtBoundSignal(float)

    achieved = (0,) * 4
    desired = (0,) * 4
    deltas = (5e-6,) * 4

    agent_name = "mild-puddle-274"

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Autonomous Beam Positioning and Focusing at ARES EA")

        vbox = qtw.QVBoxLayout()
        vbox.addWidget(self.make_agent_selection())
        vbox.addWidget(self.make_magent_setting())
        vbox.addWidget(self.make_desired_selection())
        vbox.addWidget(self.make_run_agent())
        self.setLayout(vbox)
        
        self.read_thread = AcceleratorReadThread()
        self.read_thread.screen_updated.connect(self.live_screen_view.update_screen_data)

        self.desired_updated.connect(self.update_desired_labels)
        self.desired_updated.connect(self.live_screen_view.update_target)
        self.desired_updated.connect(self.update_delta_achieved_indicators)

        self.achieved_updated.connect(self.update_achieved_labels)
        self.achieved_updated.connect(self.update_delta_achieved_indicators)

        self.deltas_updated.connect(self.update_delta_labels)
        self.deltas_updated.connect(self.update_delta_achieved_indicators)

        self.read_thread.start()

        self.achieved_updated.emit(*self.achieved)
        self.desired_updated.emit(*self.desired)
        self.deltas_updated.emit(*self.deltas)
    
    def make_agent_selection(self):
        label = qtw.QLabel("Agent")

        dropdown = qtw.QComboBox()
        dropdown.addItems([
            "mild-puddle-274", 
            "rural-salad-275", 
            "sparkling-surf-276", 
            "warm-wave-276", 
            "winter-wood-277"
        ])
        dropdown.currentTextChanged.connect(self.switch_agent)

        hbox = qtw.QHBoxLayout()
        hbox.addWidget(label)
        hbox.addWidget(dropdown)

        group_box = qtw.QGroupBox("Select Agent")
        group_box.setLayout(hbox)

        return group_box
    
    def make_magent_setting(self):
        self.magnet_dropdown = qtw.QComboBox()
        self.magnet_dropdown.addItems([
            "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH",
            "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH",
            "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH",
            "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD",
            "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD"
        ])
        self.magnet_dropdown.currentTextChanged.connect(self.read_magnet)

        self.magnet_value_field = qtw.QLineEdit()

        magnet_write_button = qtw.QPushButton("write")
        magnet_write_button.clicked.connect(self.write_magnet)

        hbox = qtw.QHBoxLayout()
        hbox.addWidget(self.magnet_dropdown)
        hbox.addWidget(self.magnet_value_field)
        hbox.addWidget(magnet_write_button)

        group_box = qtw.QGroupBox("Change Magnet Settings")
        group_box.setLayout(hbox)

        return group_box
    
    def make_desired_selection(self):
        self.achieved_mu_x_label = qtw.QLabel()
        self.achieved_mu_y_label = qtw.QLabel()
        self.achieved_sigma_x_label = qtw.QLabel()
        self.achieved_sigma_y_label = qtw.QLabel()

        self.target_delta_mu_x_label = qtw.QLabel()
        self.target_delta_mu_x_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.target_delta_mu_x_slider.setRange(0, 100)
        self.target_delta_mu_x_slider.valueChanged.connect(self.update_deltas)
        self.target_delta_mu_y_label = qtw.QLabel()
        self.target_delta_mu_y_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.target_delta_mu_y_slider.setRange(0, 100)
        self.target_delta_mu_y_slider.valueChanged.connect(self.update_deltas)
        self.target_delta_sigma_x_label = qtw.QLabel()
        self.target_delta_sigma_x_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.target_delta_sigma_x_slider.setRange(0, 100)
        self.target_delta_sigma_x_slider.valueChanged.connect(self.update_deltas)
        self.target_delta_sigma_y_label = qtw.QLabel()
        self.target_delta_sigma_y_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.target_delta_sigma_y_slider.setRange(0, 100)
        self.target_delta_sigma_y_slider.valueChanged.connect(self.update_deltas)

        self.desired_mu_x_label = qtw.QLabel()
        self.desired_mu_y_label = qtw.QLabel()
        self.desired_sigma_x_label = qtw.QLabel()
        self.desired_sigma_y_label = qtw.QLabel()

        self.mu_x_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.mu_x_slider.setRange(-50, 50)
        self.mu_x_slider.valueChanged.connect(self.update_desired)

        self.mu_y_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.mu_y_slider.setRange(-50, 50)
        self.mu_y_slider.valueChanged.connect(self.update_desired)

        self.sigma_x_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.sigma_x_slider.setRange(0, 100)
        self.sigma_x_slider.valueChanged.connect(self.update_desired)

        self.sigma_y_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.sigma_y_slider.setRange(0, 100)
        self.sigma_y_slider.valueChanged.connect(self.update_desired)
        
        grid = qtw.QGridLayout()
        grid.addWidget(self.achieved_mu_x_label, 0, 0, 1, 1)
        grid.addWidget(self.achieved_mu_y_label, 0, 2, 1, 1)
        grid.addWidget(self.achieved_sigma_x_label, 0, 4, 1, 1)
        grid.addWidget(self.achieved_sigma_y_label, 0, 6, 1, 1)
        grid.addWidget(self.target_delta_mu_x_label, 1, 0, 1, 1)
        grid.addWidget(self.target_delta_mu_x_slider, 1, 1, 1, 1)
        grid.addWidget(self.target_delta_mu_y_label, 1, 2, 1, 1)
        grid.addWidget(self.target_delta_mu_y_slider, 1, 3, 1, 1)
        grid.addWidget(self.target_delta_sigma_x_label, 1, 4, 1, 1)
        grid.addWidget(self.target_delta_sigma_x_slider, 1, 5, 1, 1)
        grid.addWidget(self.target_delta_sigma_y_label, 1, 6, 1, 1)
        grid.addWidget(self.target_delta_sigma_y_slider, 1, 7, 1, 1)
        grid.addWidget(self.desired_mu_x_label, 2, 0, 1, 1)
        grid.addWidget(self.mu_x_slider, 2, 1, 1, 1)
        grid.addWidget(self.desired_mu_y_label, 2, 2, 1, 1)
        grid.addWidget(self.mu_y_slider, 2, 3, 1, 1)
        grid.addWidget(self.desired_sigma_x_label, 2, 4, 1, 1)
        grid.addWidget(self.sigma_x_slider, 2, 5, 1, 1)
        grid.addWidget(self.desired_sigma_y_label, 2, 6, 1, 1)
        grid.addWidget(self.sigma_y_slider, 2, 7, 1, 1)

        group_box = qtw.QGroupBox("Choose Desired Beam Parameters")
        group_box.setLayout(grid)

        return group_box
    
    def make_run_agent(self):
        self.agent_screen_view = AgentScreenView((2448,2040), (3.3198e-6,2.4469e-6))

        self.progress_bar = qtw.QProgressBar()
        self.progress_bar.setMaximum(50)
        self.progress_bar.setValue(50)

        self.start_agent_button = qtw.QPushButton("Start Agent")
        self.start_agent_button.clicked.connect(self.start_agent)

        self.live_screen_view = LiveScreenView((2448,2040), (3.3198e-6,2.4469e-6))

        grid = qtw.QGridLayout()
        grid.addWidget(self.agent_screen_view, 0, 0, 1, 1)
        grid.addWidget(self.live_screen_view, 0, 1, 1, 1)
        grid.addWidget(self.progress_bar, 1, 0, 1, 2)
        grid.addWidget(self.start_agent_button, 2, 0, 1, 2)

        group_box = qtw.QGroupBox("Watch Magic Happen")
        group_box.setLayout(grid)

        return group_box
    
    @qtc.pyqtSlot(str)
    def switch_agent(self, agent_name):
        self.agent_name = agent_name
    
    @qtc.pyqtSlot()
    def update_desired(self):
        self.desired = (
            self.mu_x_slider.value() / 50 * 2.5e-3,
            self.mu_y_slider.value() / 50 * 2.0e-3,
            self.sigma_x_slider.value() / 100 * 1.0e-3,
            self.sigma_y_slider.value() / 100 * 1.0e-3
        )

        self.desired_updated.emit(*self.desired)
    
    @qtc.pyqtSlot(float, float, float, float)
    def update_desired_labels(self, mu_x, mu_y, sigma_x, sigma_y):
        mu_x, mu_y, sigma_x, sigma_y = [x * 1e3 for x in (mu_x, mu_y, sigma_x, sigma_y)]

        self.desired_mu_x_label.setText(f"µ\u2093' = {mu_x:4.3f} mm")
        self.desired_mu_y_label.setText(f"µ_y' = {mu_y:4.3f} mm")
        self.desired_sigma_x_label.setText(f"σ\u2093' = {sigma_x:4.3f} mm")
        self.desired_sigma_y_label.setText(f"σ_y' = {sigma_y:4.3f} mm")
    
    def start_agent(self):
        self.agent_thread = AgentThread(self.agent_name,
                                        np.array(self.desired),
                                        np.array(self.deltas))

        self.agent_thread.agent_screen_updated.connect(self.agent_screen_view.update_screen_data)
        self.agent_thread.took_step.connect(self.progress_bar.setValue)
        self.agent_thread.achieved_updated.connect(self.update_achieved)
        self.agent_thread.achieved_updated.connect(self.agent_screen_view.update_achieved)
        self.agent_thread.desired_updated.connect(self.agent_screen_view.update_desired)
        self.agent_thread.want_step_permission.connect(self.step_permission_prompt)
        self.agent_thread.done.connect(self.agent_finished_popup)

        self.agent_thread.start()
    
    @qtc.pyqtSlot(np.ndarray, np.ndarray)
    def step_permission_prompt(self, old_actuators, new_actuators):
        old_actuators *= [1, 1, 1, 1e3, 1e3]
        new_actuators *= [1, 1, 1, 1e3, 1e3]

        query = f"Do you allow the next step ?\n" + \
                f"\n" + \
                f"AREAMQZM1: {old_actuators[0]:+6.3f} 1/m^2 -> {new_actuators[0]:+6.3f} 1/m^2\n" + \
                f"AREAMQZM2: {old_actuators[1]:+6.3f} 1/m^2 -> {new_actuators[1]:+6.3f} 1/m^2\n" + \
                f"AREAMCVM1: {old_actuators[3]:+6.3f} mrad  -> {new_actuators[3]:+6.3f} mrad\n" + \
                f"AREAMQZM3: {old_actuators[2]:+6.3f} 1/m^2 -> {new_actuators[2]:+6.3f} 1/m^2\n" + \
                f"AREAMCHM1: {old_actuators[4]:+6.3f} mrad  -> {new_actuators[4]:+6.3f} mrad"

        answer = qtw.QMessageBox.question(
            self,
            "Step Permission",
            query,
            qtw.QMessageBox.Yes | qtw.QMessageBox.No
        )

        self.agent_thread.step_permission = (answer == qtw.QMessageBox.Yes)
        self.agent_thread.step_permission_event.set()

    @qtc.pyqtSlot(float, float, float, float)
    def update_achieved(self, mu_x, mu_y, sigma_x, sigma_y):
        self.achieved = (mu_x, mu_y, sigma_x, sigma_y)
        self.achieved_updated.emit(*self.achieved)
    
    @qtc.pyqtSlot(float, float, float, float)
    def update_achieved_labels(self, mu_x, mu_y, sigma_x, sigma_y):
        mu_x, mu_y, sigma_x, sigma_y = [x * 1e3 for x in (mu_x, mu_y, sigma_x, sigma_y)]

        self.achieved_mu_x_label.setText(f"µ\u2093 = {mu_x:4.3f} mm")
        self.achieved_mu_y_label.setText(f"µ_y = {mu_y:4.3f} mm")
        self.achieved_sigma_x_label.setText(f"σ\u2093 = {sigma_x:4.3f} mm")
        self.achieved_sigma_y_label.setText(f"σ_y = {sigma_y:4.3f} mm")
    
    @qtc.pyqtSlot()
    def update_deltas(self):
        self.deltas = (
            self.target_delta_mu_x_slider.value() / 100 * 0.5 * 1e-3,
            self.target_delta_mu_y_slider.value() / 100 * 0.5 * 1e-3,
            self.target_delta_sigma_x_slider.value() / 100 * 0.5 * 1e-3,
            self.target_delta_sigma_y_slider.value() / 100 * 0.5 * 1e-3
        )

        self.deltas_updated.emit(*self.deltas)
    
    @qtc.pyqtSlot(float, float, float, float)
    def update_delta_labels(self, mu_x, mu_y, sigma_x, sigma_y):
        mu_x, mu_y, sigma_x, sigma_y = [x * 1e3 for x in (mu_x, mu_y, sigma_x, sigma_y)]

        self.target_delta_mu_x_label.setText(f"Δµ\u2093' = {mu_x:4.3f} mm")
        self.target_delta_mu_y_label.setText(f"Δµ_y' = {mu_y:4.3f} mm")
        self.target_delta_sigma_x_label.setText(f"Δσ\u2093' = {sigma_x:4.3f} mm")
        self.target_delta_sigma_y_label.setText(f"Δσ_y' = {sigma_y:4.3f} mm")
    
    @qtc.pyqtSlot()
    def update_delta_achieved_indicators(self):
        green = "background-color: green"
        transparent = "background-color: rgba(0,0,0,0%)"

        mu_x_success = abs(self.desired[0] - self.achieved[0]) <= self.deltas[0]
        self.target_delta_mu_x_label.setStyleSheet(green if mu_x_success else transparent)

        mu_y_success = abs(self.desired[1] - self.achieved[1]) <= self.deltas[1]
        self.target_delta_mu_y_label.setStyleSheet(green if mu_y_success else transparent)

        sigma_x_success = abs(self.desired[2] - self.achieved[2]) <= self.deltas[2]
        self.target_delta_sigma_x_label.setStyleSheet(green if sigma_x_success else transparent)

        sigma_y_success = abs(self.desired[3] - self.achieved[3]) <= self.deltas[3]
        self.target_delta_sigma_y_label.setStyleSheet(green if sigma_y_success else transparent)

    @qtc.pyqtSlot(int, np.ndarray)
    def agent_finished_popup(self, steps, deltas):
        if (deltas <= self.deltas).all():
            msg = "The desired beam parameters have been achieved successfully! 🎉🎆"
        else:
            msg = "The agent timed out, the desired beam parameters cannot be achieved. 🥺"

        deltas *= 1e3
        msg += f"\n\n" + \
               f"Report:\n" + \
               f"Steps = {steps:d}\n" + \
               f"Δµ\u2093 = {deltas[0]:+6.3f} mm\n" + \
               f"Δµ_y = {deltas[1]:+6.3f} mm\n" + \
               f"Δσ\u2093 = {deltas[2]:+6.3f} mm\n" + \
               f"Δσ_y = {deltas[3]:+6.3f} mm"

        qtw.QMessageBox.information(self, "Agent Finished", msg)
    
    @qtc.pyqtSlot(str)
    def read_magnet(self, channel):
        response = pydoocs.read(channel + ".RBV")
        value = response["data"]
        self.magnet_value_field.setText(str(value))
    
    @qtc.pyqtSlot()
    def write_magnet(self):
        channel = self.magnet_dropdown.currentText()
        value = float(self.magnet_value_field.text())
        pydoocs.write(channel + ".SP", value)
        
    def handle_application_exit(self):
        pass
    

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)

    # Force the style to be the same on all OSs
    app.setStyle("Fusion")

    # Now use a palette to switch to dark colors
    palette = qtg.QPalette()
    palette.setColor(qtg.QPalette.Window, qtg.QColor(53, 53, 53))
    palette.setColor(qtg.QPalette.WindowText, qtc.Qt.white)
    palette.setColor(qtg.QPalette.Base, qtg.QColor(25, 25, 25))
    palette.setColor(qtg.QPalette.AlternateBase, qtg.QColor(53, 53, 53))
    palette.setColor(qtg.QPalette.ToolTipBase, qtc.Qt.white)
    palette.setColor(qtg.QPalette.ToolTipText, qtc.Qt.white)
    palette.setColor(qtg.QPalette.Text, qtc.Qt.white)
    palette.setColor(qtg.QPalette.Button, qtg.QColor(53, 53, 53))
    palette.setColor(qtg.QPalette.ButtonText, qtc.Qt.white)
    palette.setColor(qtg.QPalette.BrightText, qtc.Qt.red)
    palette.setColor(qtg.QPalette.Link, qtg.QColor(42, 130, 218))
    palette.setColor(qtg.QPalette.Highlight, qtg.QColor(42, 130, 218))
    palette.setColor(qtg.QPalette.HighlightedText, qtc.Qt.black)
    app.setPalette(palette)

    window = App()
    window.show()

    app.aboutToQuit.connect(window.handle_application_exit)

    sys.exit(app.exec_())
