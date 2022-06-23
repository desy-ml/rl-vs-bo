from datetime import datetime
import glob
import importlib
import math
import os
import pickle
import sys
import time

from gym.wrappers import FlattenObservation
import numpy as np
import PyQt6.QtCore as qtc
import PyQt6.QtGui as qtg
import PyQt6.QtWidgets as qtw
import pyqtgraph as pg
from stable_baselines3 import PPO, TD3
import torch


pydoocs = importlib.import_module(os.getenv("EARLMCP", "dummypydoocs"))


class ScreenView(pg.GraphicsLayoutWidget):

    user_moved_desired_ellipse = qtc.pyqtSignal(float, float, float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        resolution = (2448, 2040)
        pixel_size = (3.3198e-6, 2.4469e-6)
        self.screen_rect = (
            -resolution[0] * pixel_size[0] / 2 * 1e3,
            -resolution[1] * pixel_size[1] / 2 * 1e3,
            resolution[0] * pixel_size[0] * 1e3,
            resolution[1] * pixel_size[1] * 1e3
        )

        self.i1 = pg.ImageItem()
        self.update_agent_view(np.zeros((resolution[1], resolution[0])))

        self.desired_ellipse = pg.EllipseROI((0,0), (0,0), pen=pg.mkPen("w",width=2),
                                             rotatable=False, movable=False, resizable=False)
        # self.desired_ellipse.sigRegionChanged.connect(self.desired_ellipse_interaction)
        self.was_desired_set_programmatically = False

        self.p1 = self.addPlot(row=0, col=0, colspan=1, title="Agent View")
        self.p1.addItem(self.i1)
        self.p1.addItem(self.desired_ellipse)
        self.p1.setMouseEnabled(x=False, y=False)
        self.p1.setRange(
            xRange=(self.screen_rect[0],self.screen_rect[0]+self.screen_rect[2]),
            yRange=(self.screen_rect[1],self.screen_rect[1]+self.screen_rect[3]),
            padding=0
        )
        self.p1.getAxis("bottom").setLabel("x (mm)")
        self.p1.getAxis("left").setLabel("y (mm)")
        self.p1.hideButtons()
        self.p1.setAspectLocked()

        for handle in self.desired_ellipse.getHandles():
            self.desired_ellipse.removeHandle(handle)

        self.i2 = pg.ImageItem()
        self.update_live_view(np.zeros((resolution[1], resolution[0])))

        self.p2 = self.addPlot(row=0, col=1, colspan=1, title="Live View AR.EA.BSC.R.1")
        self.p2.addItem(self.i2)
        self.p2.setMouseEnabled(x=False, y=False)
        self.p2.setRange(
            xRange=(self.screen_rect[0],self.screen_rect[0]+self.screen_rect[2]),
            yRange=(self.screen_rect[1],self.screen_rect[1]+self.screen_rect[3]),
            padding=0
        )
        self.p2.getAxis("bottom").setLabel("x (mm)")
        self.p2.getAxis("left").setLabel("y (mm)")
        self.p2.hideButtons()
        self.p2.setAspectLocked()

        cmap = pg.colormap.get("CET-L9")
        bar = pg.ColorBarItem(cmap=cmap, width=10)
        max_level = pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ")["data"].max()
        bar.setLevels(low=0, high=max_level)
        bar.setImageItem([self.i1,self.i2], insert_in=self.p2)
    
    @qtc.pyqtSlot(np.ndarray)
    def update_agent_view(self, screen_data):
        self.i1.setImage(
            np.flipud(screen_data),
            axisOrder="row-major",
            rect=self.screen_rect,
            autoLevels=False
        )
    
    @qtc.pyqtSlot(np.ndarray)
    def update_live_view(self, screen_data):
        self.i2.setImage(
            np.flipud(screen_data),
            axisOrder="row-major",
            rect=self.screen_rect,
            autoLevels=False
        )
    
    @qtc.pyqtSlot(float, float, float, float)
    def move_desired_ellipse(self, mu_x, mu_y, sigma_x, sigma_y):
        mu_x, mu_y, sigma_x, sigma_y = [x * 1e3 for x in (mu_x, mu_y, sigma_x, sigma_y)]

        self.desired_ellipse.setPos((mu_x-sigma_x, mu_y-sigma_y))
        self.desired_ellipse.setSize((2*sigma_x, 2*sigma_y))

        self.was_desired_set_programmatically = True
    
    @qtc.pyqtSlot(float, float, float, float)
    def move_achieved_ellipse(self, mu_x, mu_y, sigma_x, sigma_y):
        if math.isnan(mu_x):
            return

        mu_x, mu_y, sigma_x, sigma_y = [x * 1e3 for x in (mu_x, mu_y, sigma_x, sigma_y)]

        x, y = mu_x - sigma_x, mu_y - sigma_y
        w, h = 2 * sigma_x, 2 * sigma_y

        if not hasattr(self, "achieved_ellipse"):
            self.achieved_ellipse = pg.EllipseROI(
                (x, y),
                (w, h),
                pen=pg.mkPen("r", width=2),
                rotatable=False,
                movable=False,
                resizable=False
            )
            self.p1.addItem(self.achieved_ellipse)
            for handle in self.achieved_ellipse.getHandles():
                self.achieved_ellipse.removeHandle(handle)
        else:
            self.achieved_ellipse.setPos((x, y))
            self.achieved_ellipse.setSize((w, h))


class MeasureBeamThread(qtc.QThread):

    agent_screen_updated = qtc.pyqtSignal(np.ndarray)
    achieved_updated = qtc.pyqtSignal(float, float, float, float)

    def run(self):
        accelerator = machine.ExperimentalArea()
        screen_data = accelerator.capture_clean_beam()
        beam_parameters = utils.compute_beam_parameters(
            screen_data,
            accelerator.pixel_size * accelerator.binning
        )

        self.agent_screen_updated.emit(screen_data)
        self.achieved_updated.emit(*beam_parameters)


class AgentThread(qtc.QThread):
    
    started = qtc.pyqtSignal()
    done = qtc.pyqtSignal(tuple)
    agent_screen_updated = qtc.pyqtSignal(np.ndarray)
    took_step = qtc.pyqtSignal(int)
    desired_updated = qtc.pyqtSignal(float, float, float, float)
    achieved_updated = qtc.pyqtSignal(float, float, float, float)
    done = qtc.pyqtSignal(int, np.ndarray)
    actuators_updated = qtc.pyqtSignal(float, float, float, float, float)

    def __init__(self, model_name, desired_goal, target_delta, experiment_name):
        super().__init__()

        self.model_name = model_name
        self.desired_goal = desired_goal
        self.target_delta = target_delta
        self.experiment_name = experiment_name

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        self.stop_requested = False

    def run(self):
        self.started.emit()
        self.took_step.emit(0)
        self.desired_updated.emit(*self.desired_goal)

        self.env = ARESEASequential(backend="machine")
        self.env = FlattenObservation(self.env)

        self.actuators_updated.emit(*self.env.accelerator.actuators)
        
        self.env.unwrapped.target_delta = self.target_delta

        model = TD3.load(f"models/{self.model_name}")

        done = False
        i = 0
        observation = self.env.reset(goal=self.desired_goal)
        log = {
            # "background": [self.env.unwrapped.background],    # TODO: Reintroduce
            # "beam": [self.env.unwrapped.beam],    # TODO: Reintroduce
            "screen_data": [self.env.unwrapped.screen_data],
            "observation": [self.env.unwrapped.observation],
            "action": [],
            "time": [time.time()]
        }
        self.log_channels(log, auxiliary_channels)
        self.agent_screen_updated.emit(self.env.screen_data)
        self.achieved_updated.emit(*self.env.unwrapped.observation["achieved_goal"])
        while not done:
            action, _ = model.predict(observation)

            observation, _, done, _ = self.env.step(action)

            # log["background"].append(self.env.unwrapped.background)   # TODO: Reintroduce
            # log["beam"].append(self.env.unwrapped.beam)   # TODO: Reintroduce
            log["screen_data"].append(self.env.unwrapped.screen_data)
            log["observation"].append(self.env.unwrapped.observation)
            log["action"].append(self.env.unwrapped.action2accelerator(action))
            self.log_channels(log, auxiliary_channels)
            log["time"].append(time.time())

            self.actuators_updated.emit(*self.env.accelerator.actuators)
            self.agent_screen_updated.emit(self.env.unwrapped.screen_data)
            self.achieved_updated.emit(*self.env.unwrapped.observation["achieved_goal"])
            i += 1
            self.took_step.emit(i)

            print(f"Finished step {i}")

            if self.stop_requested:
                break
        
        self.took_step.emit(50)

        self.env.close()

        log["history"] = self.env.unwrapped.history
        log["model_name"] = self.model_name
        log["target_delta"] = self.env.unwrapped.target_delta

        logpath = f"experiments/{self.timestamp}_{self.experiment_name}/"
        os.mkdir(logpath)

        logfilename = logpath + "log.pkl"
        with open(logfilename, "wb") as f:
            pickle.dump(log, f)
            print(f"Log file saved as \"{logfilename}\"")
        
        desired = self.env.unwrapped.observation["desired_goal"]
        achieved = self.env.unwrapped.observation["achieved_goal"]
        delta = np.abs(desired - achieved)
        self.done.emit(i, delta)
    
    def log_channels(self, log, channels):
        for channel in channels:
            sample = pydoocs.read(channel)["data"]
            if not channel in log.keys():
                log[channel] = sample
            elif not isinstance(log[channel], list):
                log[channel] = [log[channel], sample]
            else:
                log[channel].append(sample)


class App(qtw.QWidget):

    achieved_updated = qtc.pyqtSignal(float, float, float, float)
    desired_updated = qtc.pyqtSignal(float, float, float, float)
    deltas_updated = qtc.pyqtSignal(float, float, float, float)

    achieved = (float("nan"),) * 4
    desired = [0, 0, 1e-5, 1e-5]
    deltas = (5e-6,) * 4

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Autonomous Beam Positioning and Focusing at ARES EA")

        vbox = qtw.QVBoxLayout()
        vbox.addWidget(self.make_rl_setup())
        vbox.addWidget(self.make_measure_beam())   # TODO Make measure beam
        vbox.addWidget(self.make_desired_selection())
        vbox.addWidget(self.make_run_agent())
        self.setLayout(vbox)

        self.desired_updated.connect(self.update_desired_labels)
        self.desired_updated.connect(self.screen_view.move_desired_ellipse)
        self.desired_updated.connect(self.update_delta_achieved_indicators)

        self.achieved_updated.connect(self.update_achieved_labels)
        self.achieved_updated.connect(self.screen_view.move_achieved_ellipse)
        self.achieved_updated.connect(self.update_delta_achieved_indicators)

        self.deltas_updated.connect(self.update_delta_labels)
        self.deltas_updated.connect(self.update_delta_achieved_indicators)

        self.achieved_updated.emit(*self.achieved)
        self.desired_updated.emit(*self.desired)
        self.deltas_updated.emit(*self.deltas)

        self.resize(1200, 800)
    
    def make_measure_beam(self):
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
        self.magnet_value_field.setMaximumWidth(100)

        self.magnet_write_button = qtw.QPushButton("write")
        self.magnet_write_button.clicked.connect(self.write_magnet)

        self.measure_beam_button = qtw.QPushButton("measure beam")
        self.measure_beam_button.setMinimumWidth(135)
        self.measure_beam_button.clicked.connect(self.measure_beam)

        hbox = qtw.QHBoxLayout()
        hbox.addWidget(self.magnet_dropdown)
        hbox.addWidget(self.magnet_value_field)
        hbox.addWidget(self.magnet_write_button)
        hbox.addStretch()
        hbox.addWidget(self.measure_beam_button)

        group_box = qtw.QGroupBox("1. Measure beam TODO move current beam values here")
        group_box.setLayout(hbox)

        return group_box
    
    def make_desired_selection(self):
        self.achieved_mu_x_label = qtw.QLabel()
        self.achieved_mu_y_label = qtw.QLabel()
        self.achieved_sigma_x_label = qtw.QLabel()
        self.achieved_sigma_y_label = qtw.QLabel()

        self.target_delta_mu_x_label = qtw.QLabel()
        self.target_delta_mu_x_slider = qtw.QSlider()
        self.target_delta_mu_x_slider.setRange(0, 100)
        self.target_delta_mu_x_slider.valueChanged.connect(self.update_deltas)
        self.target_delta_mu_y_label = qtw.QLabel()
        self.target_delta_mu_y_slider = qtw.QSlider()
        self.target_delta_mu_y_slider.setRange(0, 100)
        self.target_delta_mu_y_slider.valueChanged.connect(self.update_deltas)
        self.target_delta_sigma_x_label = qtw.QLabel()
        self.target_delta_sigma_x_slider = qtw.QSlider()
        self.target_delta_sigma_x_slider.setRange(0, 100)
        self.target_delta_sigma_x_slider.valueChanged.connect(self.update_deltas)
        self.target_delta_sigma_y_label = qtw.QLabel()
        self.target_delta_sigma_y_slider = qtw.QSlider()
        self.target_delta_sigma_y_slider.setRange(0, 100)
        self.target_delta_sigma_y_slider.valueChanged.connect(self.update_deltas)

        self.desired_mu_x_label = qtw.QLabel()
        self.desired_mu_y_label = qtw.QLabel()
        self.desired_sigma_x_label = qtw.QLabel()
        self.desired_sigma_y_label = qtw.QLabel()

        self.mu_x_slider = qtw.QSlider()
        self.mu_x_slider.setRange(-50, 50)
        self.mu_x_slider.valueChanged.connect(self.update_desired)

        self.mu_y_slider = qtw.QSlider()
        self.mu_y_slider.setRange(-50, 50)
        self.mu_y_slider.valueChanged.connect(self.update_desired)

        self.sigma_x_slider = qtw.QSlider()
        self.sigma_x_slider.setRange(1, 100)
        self.sigma_x_slider.valueChanged.connect(self.update_desired)

        self.sigma_y_slider = qtw.QSlider()
        self.sigma_y_slider.setRange(1, 100)
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

        group_box = qtw.QGroupBox("2. Choose desired beam parameters")
        group_box.setLayout(grid)

        return group_box
    
    def make_rl_setup(self):
        model_files = glob.glob("models/*/*-*-*.zip") + glob.glob("models/onestep_vpg/onestep-model-*.pkl")
        models = [filename[7:-4] for filename in model_files]
        models.append("Bayesian Optimisation")
        models = sorted(models)

        self.agent_name = models[0]

        label1 = qtw.QLabel("Agent")

        self.model_dropdown = qtw.QComboBox()
        self.model_dropdown.addItems(models)
        self.model_dropdown.currentTextChanged.connect(self.switch_agent)

        label2 = qtw.QLabel("Experiment name")

        self.experiment_name_field = qtw.QLineEdit()

        hbox = qtw.QHBoxLayout()
        hbox.addWidget(label1)
        hbox.addWidget(self.model_dropdown)
        hbox.addStretch()
        hbox.addWidget(label2)
        hbox.addWidget(self.experiment_name_field)
        hbox.addStretch()

        group_box = qtw.QGroupBox("TODO move out of box and center selection")
        group_box.setLayout(hbox)

        return group_box
    
    def make_run_agent(self):
        self.screen_view = ScreenView()
        # self.screen_view.user_moved_desired_ellipse.connect(self.update_desired)

        self.start_stop_button = qtw.QPushButton("Optimise")
        self.start_stop_button.setStyleSheet("background-color: darkgreen")
        self.start_stop_button.clicked.connect(self.start_agent)
        self.experiment_name_field.textChanged.connect(self.check_is_start_agent_allowed)
        self.check_is_start_agent_allowed(self.experiment_name_field.text())

        grid = qtw.QGridLayout()
        grid.addWidget(self.screen_view, 0, 0, 1, 2)
        grid.addWidget(self.start_stop_button, 1, 0, 1, 2)

        group_box = qtw.QGroupBox("4. TODO dont have this in a box?")
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
    
    @qtc.pyqtSlot()
    def measure_beam(self):
        self.measure_beam_thread = MeasureBeamThread()
        self.measure_beam_thread.agent_screen_updated.connect(self.screen_view.update_agent_view)
        self.measure_beam_thread.achieved_updated.connect(self.update_achieved)
        self.measure_beam_thread.run()
    
    def start_agent(self):
        self.agent_thread = AgentThread(self.agent_name,
                                        np.array(self.desired),
                                        np.array(self.deltas),
                                        self.experiment_name_field.text())

        self.agent_thread.agent_screen_updated.connect(self.screen_view.update_agent_view)
        self.agent_thread.achieved_updated.connect(self.update_achieved)
        self.agent_thread.done.connect(self.agent_finished_popup)
        self.agent_thread.started.connect(self.stop_user_interaction)
        self.agent_thread.done.connect(self.start_user_interaction)

        self.actuator_plot.reset()
        self.agent_thread.actuators_updated.connect(self.actuator_plot.update)

        self.agent_thread.start()
    
    def stop_agent(self):
        self.agent_thread.stop_requested = True
    
    def set_user_interaction_allowed(self, is_allowed):
        self.magnet_dropdown.setEnabled(is_allowed)
        self.magnet_value_field.setEnabled(is_allowed)
        self.magnet_write_button.setEnabled(is_allowed)
        self.measure_beam_button.setEnabled(is_allowed)

        self.target_delta_mu_x_slider.setEnabled(is_allowed)
        self.target_delta_mu_y_slider.setEnabled(is_allowed)
        self.target_delta_sigma_x_slider.setEnabled(is_allowed)
        self.target_delta_sigma_y_slider.setEnabled(is_allowed)
        self.mu_x_slider.setEnabled(is_allowed)
        self.mu_y_slider.setEnabled(is_allowed)
        self.sigma_x_slider.setEnabled(is_allowed)
        self.sigma_y_slider.setEnabled(is_allowed)

        self.model_dropdown.setEnabled(is_allowed)
        self.experiment_name_field.setEnabled(is_allowed)
    
    def stop_user_interaction(self):
        self.set_user_interaction_allowed(False)

        self.start_stop_button.clicked.disconnect(self.start_agent)
        self.start_stop_button.clicked.connect(self.stop_agent)
        self.start_stop_button.setText("Stop")
        self.start_stop_button.setStyleSheet("background-color: darkred")
    
    def start_user_interaction(self):
        self.set_user_interaction_allowed(True)

        self.start_stop_button.clicked.disconnect(self.stop_agent)
        self.start_stop_button.clicked.connect(self.start_agent)
        self.start_stop_button.setText("Optimise")
        self.start_stop_button.setStyleSheet("background-color: darkgreen")
    
    @qtc.pyqtSlot(str)
    def check_is_start_agent_allowed(self, experiment_name):
        self.start_stop_button.setEnabled(len(experiment_name) > 0)

    @qtc.pyqtSlot(float, float, float, float)
    def update_achieved(self, mu_x, mu_y, sigma_x, sigma_y):
        self.achieved = (mu_x, mu_y, sigma_x, sigma_y)
        self.achieved_updated.emit(*self.achieved)
    
    @qtc.pyqtSlot(float, float, float, float)
    def update_achieved_labels(self, mu_x, mu_y, sigma_x, sigma_y):
        if not math.isnan(mu_x):
            mu_x, mu_y, sigma_x, sigma_y = [x * 1e3 for x in (mu_x, mu_y, sigma_x, sigma_y)]

            self.achieved_mu_x_label.setText(f"µ\u2093 = {mu_x:4.3f} mm")
            self.achieved_mu_y_label.setText(f"µ_y = {mu_y:4.3f} mm")
            self.achieved_sigma_x_label.setText(f"σ\u2093 = {sigma_x:4.3f} mm")
            self.achieved_sigma_y_label.setText(f"σ_y = {sigma_y:4.3f} mm")
        else:
            self.achieved_mu_x_label.setText(f"µ\u2093 = -")
            self.achieved_mu_y_label.setText(f"µ_y = -")
            self.achieved_sigma_x_label.setText(f"σ\u2093 = -")
            self.achieved_sigma_y_label.setText(f"σ_y = -")
    
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

        known = not math.isnan(self.achieved[0])

        mu_x_success = known and abs(self.desired[0] - self.achieved[0]) <= self.deltas[0]
        self.target_delta_mu_x_label.setStyleSheet(green if mu_x_success else transparent)

        mu_y_success = known and abs(self.desired[1] - self.achieved[1]) <= self.deltas[1]
        self.target_delta_mu_y_label.setStyleSheet(green if mu_y_success else transparent)

        sigma_x_success = known and abs(self.desired[2] - self.achieved[2]) <= self.deltas[2]
        self.target_delta_sigma_x_label.setStyleSheet(green if sigma_x_success else transparent)

        sigma_y_success = known and abs(self.desired[3] - self.achieved[3]) <= self.deltas[3]
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
    # palette = qtg.QPalette()
    # palette.setColor(qtg.QPalette.Window, qtg.QColor(53, 53, 53))
    # palette.setColor(qtg.QPalette.WindowText, qtc.Qt.white)
    # palette.setColor(qtg.QPalette.Base, qtg.QColor(25, 25, 25))
    # palette.setColor(qtg.QPalette.AlternateBase, qtg.QColor(53, 53, 53))
    # palette.setColor(qtg.QPalette.ToolTipBase, qtc.Qt.white)
    # palette.setColor(qtg.QPalette.ToolTipText, qtc.Qt.white)
    # palette.setColor(qtg.QPalette.Text, qtc.Qt.white)
    # palette.setColor(qtg.QPalette.Button, qtg.QColor(53, 53, 53))
    # palette.setColor(qtg.QPalette.ButtonText, qtc.Qt.white)
    # palette.setColor(qtg.QPalette.BrightText, qtc.Qt.red)
    # palette.setColor(qtg.QPalette.Link, qtg.QColor(42, 130, 218))
    # palette.setColor(qtg.QPalette.Highlight, qtg.QColor(42, 130, 218))
    # palette.setColor(qtg.QPalette.HighlightedText, qtc.Qt.black)
    # app.setPalette(palette)

    window = App()
    window.show()

    app.aboutToQuit.connect(window.handle_application_exit)

    sys.exit(app.exec())
