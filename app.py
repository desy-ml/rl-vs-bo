import sys
from time import sleep

from gym.wrappers import FlattenObservation, TimeLimit
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

        self.setFixedSize(600, 450)
    
    def create_plot(self):
        self.screen_plot = self.ax.imshow(
            np.zeros(self.resolution), 
            cmap="magma",
            interpolation="None",
            extent=self.screen_extent
        )
        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("y (mm)")

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

        self.setFixedSize(600, 450)
    
    def create_plot(self):
        self.screen_plot = self.ax.imshow(
            np.zeros(self.resolution), 
            cmap="magma",
            interpolation="None",
            extent=self.screen_extent
        )
        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("y (mm)")

        self.fig.tight_layout()
    
    @qtc.pyqtSlot(np.ndarray)
    def update_screen_data(self, screen_data):
        self.screen_plot.set_data(screen_data)
        self.screen_plot.set_clim(vmin=0, vmax=screen_data.max())

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
    took_step = qtc.pyqtSignal(int)

    def __init__(self, desired_goal):
        super().__init__()

        self.desired_goal = desired_goal

    def run(self):
        self.took_step.emit(0)

        env = ARESEAMachine()
        env = TimeLimit(env, max_episode_steps=50)
        env = FlattenObservation(env)

        model = TD3.load("models/pretty-jazz-258")

        done = False
        i = 0
        observation = env.reset(goal=self.desired_goal)
        self.agent_screen_updated.emit(env.screen_data)
        while not done:
            action, _ = model.predict(observation)
            observation, _, done, _ = env.step(action)
            self.agent_screen_updated.emit(env.screen_data)
            i += 1
            self.took_step.emit(i)
    
    def change_grating(self, grating):
        print(f"Changing grating to {grating}")


class App(qtw.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Autonomous Beam Position and Focusing at ARES EA")

        self.agent_screen_view = AgentScreenView((2448,2040), (3.3198e-6,2.4469e-6))

        self.progress_bar = qtw.QProgressBar()
        self.progress_bar.setMaximum(50)
        self.progress_bar.setValue(50)

        self.start_agent_button = qtw.QPushButton("Start Agent")
        self.start_agent_button.clicked.connect(self.start_agent)

        self.live_screen_view = LiveScreenView((2448,2040), (3.3198e-6,2.4469e-6))
        
        self.read_thread = AcceleratorReadThread()
        self.read_thread.screen_updated.connect(self.live_screen_view.update_screen_data)

        self.mu_x_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.mu_x_slider.setRange(-50, 50)
        self.mu_x_slider.valueChanged.connect(self.live_screen_view.select_mu_x)

        self.mu_y_slider = qtw.QSlider(qtc.Qt.Vertical)
        self.mu_y_slider.setRange(-50, 50)
        self.mu_y_slider.valueChanged.connect(self.live_screen_view.select_mu_y)

        self.sigma_x_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.sigma_x_slider.setRange(0, 100)
        self.sigma_x_slider.valueChanged.connect(self.live_screen_view.select_sigma_x)

        self.sigma_y_slider = qtw.QSlider(qtc.Qt.Vertical)
        self.sigma_y_slider.setRange(0, 100)
        self.sigma_y_slider.valueChanged.connect(self.live_screen_view.select_sigma_y)

        hbox = qtw.QHBoxLayout()
        vbox = qtw.QVBoxLayout()
        vbox.addWidget(self.agent_screen_view)
        vbox.addWidget(self.progress_bar)
        vbox.addWidget(self.start_agent_button)
        vbox.addStretch()
        hbox.addLayout(vbox)
        grid = qtw.QGridLayout()
        grid.addWidget(self.live_screen_view, 0, 0)
        grid.addWidget(self.mu_x_slider, 1, 0)
        grid.addWidget(self.sigma_x_slider, 2, 0)
        grid.addWidget(self.mu_y_slider, 0, 1)
        grid.addWidget(self.sigma_y_slider, 0, 2)
        grid.setRowStretch(3, 1)
        hbox.addLayout(grid)
        self.setLayout(hbox)

        self.read_thread.start()
    
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
        self.agent_thread.start()
    
    def handle_application_exit(self):
        print("Handling application exit")
    

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
