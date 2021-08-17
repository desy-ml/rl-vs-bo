from datetime import datetime
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
    
    @qtc.pyqtSlot(int)
    def select_mu_x(self, mu_x):
        self.mu_x = mu_x / 50 * 2.5e-3 * 1e3

        self.select_ellipse.set_center((self.mu_x,self.mu_y))
        self.draw()
    
    @qtc.pyqtSlot(int)
    def select_mu_y(self, mu_y):
        self.mu_y = mu_y / 50 * 2.0e-3 * 1e3
        self.select_ellipse.set_center((self.mu_x,self.mu_y))
        self.draw()
    
    @qtc.pyqtSlot(int)
    def select_sigma_x(self, sigma_x):
        self.sigma_x = sigma_x / 100 * 1.0e-3 * 1e3
        self.select_ellipse.set_width(2 * self.sigma_x)
        self.draw()
    
    @qtc.pyqtSlot(int)
    def select_sigma_y(self, sigma_y):
        self.sigma_y = sigma_y / 100 * 1.0e-3 * 1e3
        self.select_ellipse.set_height(2 * self.sigma_y)
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
    
    @qtc.pyqtSlot(np.ndarray)
    def update_achieved_goal(self, achieved_goal):
        self.achieved_goal = achieved_goal * 1e3

        self.achieved_ellipse.set_center((self.achieved_goal[0],self.achieved_goal[1]))
        self.achieved_ellipse.set_width(2 * self.achieved_goal[2])
        self.achieved_ellipse.set_height(2 * self.achieved_goal[3])
        self.draw()
    
    @qtc.pyqtSlot(np.ndarray)
    def update_desired_goal(self, desired_goal):
        self.desired_goal = desired_goal * 1e3

        self.desired_ellipse.set_center((self.desired_goal[0],self.desired_goal[1]))
        self.desired_ellipse.set_width(2 * self.desired_goal[2])
        self.desired_ellipse.set_height(2 * self.desired_goal[3])
        self.draw()


class AcceleratorReadThread(qtc.QThread):

    screen_binning = 4

    screen_updated = qtc.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()

        pydoocs.write("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL", self.screen_binning)
        pydoocs.write("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL", self.screen_binning)

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
    desired_goal_updated = qtc.pyqtSignal(np.ndarray)
    achieved_goal_updated = qtc.pyqtSignal(np.ndarray)

    step_permission_event = Event()

    def __init__(self, desired_goal):
        super().__init__()

        self.desired_goal = desired_goal

        self.step_permission_event.clear()
        self.step_permission = False

    def run(self):
        self.took_step.emit(0)
        self.desired_goal_updated.emit(self.desired_goal)

        self.env = ARESEAMachine()
        self.env = TimeLimit(self.env, max_episode_steps=50)
        self.env = FlattenObservation(self.env)
        self.env = Monitor(self.env,
                           f"experiments/{datetime.now().strftime('%m%d%Y%H%M%S')}_recording",
                           video_callable=lambda i: True)

        model = TD3.load("models/pretty-jazz-258")

        done = False
        i = 0
        observation = self.env.reset(goal=self.desired_goal)
        self.agent_screen_updated.emit(self.env.screen_data)
        self.achieved_goal_updated.emit(self.env.unwrapped.observation["achieved_goal"])
        while not done:
            action, _ = model.predict(observation)

            if not self.ask_step_permission(action, observation):
                print("Permission denied!")
                self.took_step.emit(50)
                break
            print("Permission granted!")

            observation, _, done, _ = self.env.step(action)

            self.agent_screen_updated.emit(self.env.unwrapped.screen_data)
            self.achieved_goal_updated.emit(self.env.unwrapped.observation["achieved_goal"])
            i += 1
            self.took_step.emit(i)
        
        self.env.close()
    
    def ask_step_permission(self, action, observation):
        old_actuators = observation[-5:] * self.env.accelerator_observation_space["observation"].high[-5:]
        new_actuators = old_actuators + action * self.env.accelerator_action_space.high

        self.want_step_permission.emit(old_actuators, new_actuators)
        self.step_permission_event.wait()
        self.step_permission_event.clear()

        tmp = self.step_permission
        self.step_permission = False

        return tmp


class App(qtw.QWidget):

    achieved_beam_parameters = [0] * 4
    desired_beam_parameters = [0] * 4

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Autonomous Beam Positioning and Focusing at ARES EA")

        self.achieved_mu_x_label = qtw.QLabel()
        self.achieved_mu_y_label = qtw.QLabel()
        self.achieved_sigma_x_label = qtw.QLabel()
        self.achieved_sigma_y_label = qtw.QLabel()

        self.agent_screen_view = AgentScreenView((2448,2040), (3.3198e-6,2.4469e-6))

        self.progress_bar = qtw.QProgressBar()
        self.progress_bar.setMaximum(50)
        self.progress_bar.setValue(50)

        self.start_agent_button = qtw.QPushButton("Start Agent")
        self.start_agent_button.clicked.connect(self.start_agent)

        self.live_screen_view = LiveScreenView((2448,2040), (3.3198e-6,2.4469e-6))

        self.desired_mu_x_label = qtw.QLabel()
        self.desired_mu_y_label = qtw.QLabel()
        self.desired_sigma_x_label = qtw.QLabel()
        self.desired_sigma_y_label = qtw.QLabel()
        
        self.read_thread = AcceleratorReadThread()
        self.read_thread.screen_updated.connect(self.live_screen_view.update_screen_data)

        self.mu_x_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.mu_x_slider.setRange(-50, 50)
        self.mu_x_slider.valueChanged.connect(self.live_screen_view.select_mu_x)
        self.mu_x_slider.valueChanged.connect(self.update_beam_parameter_labels)

        self.mu_y_slider = qtw.QSlider(qtc.Qt.Vertical)
        self.mu_y_slider.setRange(-50, 50)
        self.mu_y_slider.valueChanged.connect(self.live_screen_view.select_mu_y)
        self.mu_y_slider.valueChanged.connect(self.update_beam_parameter_labels)

        self.sigma_x_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.sigma_x_slider.setRange(0, 100)
        self.sigma_x_slider.valueChanged.connect(self.live_screen_view.select_sigma_x)
        self.sigma_x_slider.valueChanged.connect(self.update_beam_parameter_labels)

        self.sigma_y_slider = qtw.QSlider(qtc.Qt.Vertical)
        self.sigma_y_slider.setRange(0, 100)
        self.sigma_y_slider.valueChanged.connect(self.live_screen_view.select_sigma_y)
        self.sigma_y_slider.valueChanged.connect(self.update_beam_parameter_labels)

        grid = qtw.QGridLayout()
        grid.addWidget(self.achieved_mu_x_label, 0, 0, 1, 1)
        grid.addWidget(self.achieved_mu_y_label, 0, 1, 1, 1)
        grid.addWidget(self.achieved_sigma_x_label, 0, 2, 1, 1)
        grid.addWidget(self.achieved_sigma_y_label, 0, 3, 1, 1)
        grid.addWidget(self.agent_screen_view, 1, 0, 1, 4)
        grid.addWidget(self.progress_bar, 2, 0, 1, 4)
        grid.addWidget(self.start_agent_button, 3, 0, 1, 4)
        grid.addWidget(self.desired_mu_x_label, 0, 4, 1, 1)
        grid.addWidget(self.desired_mu_y_label, 0, 5, 1, 1)
        grid.addWidget(self.desired_sigma_x_label, 0, 6, 1, 1)
        grid.addWidget(self.desired_sigma_y_label, 0, 7, 1, 1)
        grid.addWidget(self.live_screen_view, 1, 4, 1, 4)
        grid.addWidget(self.mu_x_slider, 2, 4, 1, 4)
        grid.addWidget(self.sigma_x_slider, 3, 4, 1, 4)
        grid.addWidget(self.mu_y_slider, 1, 8, 1, 1)
        grid.addWidget(self.sigma_y_slider, 1, 9, 1, 1)
        grid.setRowStretch(4, 1)
        self.setLayout(grid)

        self.read_thread.start()
        self.update_beam_parameter_labels()
    
    def start_agent(self):
        desired_goal = np.array([
            self.live_screen_view.mu_x,
            self.live_screen_view.mu_y,
            self.live_screen_view.sigma_x,
            self.live_screen_view.sigma_y
        ]) * 1e-3

        self.agent_thread = AgentThread(desired_goal)

        self.agent_thread.agent_screen_updated.connect(self.agent_screen_view.update_screen_data)
        self.agent_thread.took_step.connect(self.progress_bar.setValue)
        self.agent_thread.took_step.connect(self.update_beam_parameter_labels)
        self.agent_thread.achieved_goal_updated.connect(self.agent_screen_view.update_achieved_goal)
        self.agent_thread.desired_goal_updated.connect(self.agent_screen_view.update_desired_goal)
        self.agent_thread.want_step_permission.connect(self.step_permission_prompt)

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
    
    def update_beam_parameter_labels(self):
        if hasattr(self, "agent_thread") and hasattr(self.agent_thread, "env"):
            self.achieved_beam_parameters = self.agent_thread.env.unwrapped.observation["achieved_goal"] * 1e3
        self.desired_beam_parameters = [
            self.mu_x_slider.value() / 50 * 2.5e-3 * 1e3,
            self.mu_y_slider.value() / 50 * 2.0e-3 * 1e3,
            self.sigma_x_slider.value() / 100 * 1.0e-3 * 1e3,
            self.sigma_y_slider.value() / 100 * 1.0e-3 * 1e3
        ]

        self.achieved_mu_x_label.setText(f"µ\u2093 = {self.achieved_beam_parameters[0]:4.3f} mm")
        self.achieved_mu_y_label.setText(f"µ_y = {self.achieved_beam_parameters[1]:4.3f} mm")
        self.achieved_sigma_x_label.setText(f"σ\u2093 = {self.achieved_beam_parameters[2]:4.3f} mm")
        self.achieved_sigma_y_label.setText(f"σ_y = {self.achieved_beam_parameters[3]:4.3f} mm")

        self.desired_mu_x_label.setText(f"µ\u2093' = {self.desired_beam_parameters[0]:4.3f} mm")
        self.desired_mu_y_label.setText(f"µ_y' = {self.desired_beam_parameters[1]:4.3f} mm")
        self.desired_sigma_x_label.setText(f"σ\u2093' = {self.desired_beam_parameters[2]:4.3f} mm")
        self.desired_sigma_y_label.setText(f"σ_y' = {self.desired_beam_parameters[3]:4.3f} mm")
        
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
