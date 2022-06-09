import importlib
import os
import sys

import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import pyqtgraph as pg


pydoocs = importlib.import_module(os.getenv("EARLMCP", "dummypydoocs"))


def create_environment():
    def make_env():
        env = ARESEA()
        env = FilterObservation(env, ["beam","magnets"])
        env = FilterAction(env, [0,1,3], replace=0)
        env = TimeLimit(env, max_episode_steps=50)
        env = RecordVideo(env, video_folder="recordings_real")
        env = RecordData(env)
        env = FlattenObservation(env)
        env = FrameStack(env, 4)
        env = RescaleAction(env, -3, 3)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    # env = VecNormalize.load("model_env.pkl", env)
    env.training = False


def create_model():
    model = PPO.load("models/smooth-pond-130/model")


class App(qtw.QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.setup_gui()
    
    def setup_gui(self):
        self.setWindowTitle("Autonomous Beam Positioning and Focusing at ARES v2")
        
        # Create UI elements
        self.screen_select_dropdown = qtw.QComboBox()
        self.screen_select_dropdown.addItems(["AREABSCR1"])
        
        self.desired_mu_x_label = qtw.QLabel("mu_x")
        self.desired_mu_x_input = qtw.QLineEdit()
        self.desired_mu_x_input.setText("0.0")
        self.desired_sigma_x_label = qtw.QLabel("sigma_x")
        self.desired_sigma_x_input = qtw.QLineEdit()
        self.desired_sigma_x_input.setText("0.0")
        self.desired_mu_y_label = qtw.QLabel("mu_y")
        self.desired_mu_y_input = qtw.QLineEdit()
        self.desired_mu_y_input.setText("0.0")
        self.desired_sigma_y_label = qtw.QLabel("sigma_y")
        self.desired_sigma_y_input = qtw.QLineEdit()
        self.desired_sigma_y_input.setText("0.0")
        
        self.start_stop_button = qtw.QPushButton("Start")
        self.start_stop_button .setStyleSheet("background-color: darkgreen")
        
        self.measure_beam_button = qtw.QPushButton("Measure beam")
        
        self.screen_plot = ScreenPlot()
        
        # Set up layout
        vbox = qtw.QVBoxLayout()
        vbox.addWidget(self.screen_select_dropdown)
        grid = qtw.QGridLayout()
        grid.addWidget(self.desired_mu_x_label, 0, 0)
        grid.addWidget(self.desired_mu_x_input, 0, 1)
        grid.addWidget(self.desired_sigma_x_label, 0, 2)
        grid.addWidget(self.desired_sigma_x_input, 0, 3)
        grid.addWidget(self.desired_mu_y_label, 1, 0)
        grid.addWidget(self.desired_mu_y_input, 1, 1)
        grid.addWidget(self.desired_sigma_y_label, 1, 2)
        grid.addWidget(self.desired_sigma_y_input, 1, 3)
        vbox.addLayout(grid)
        vbox.addWidget(self.start_stop_button)
        vbox.addWidget(self.measure_beam_button)
        vbox.addWidget(self.screen_plot)
        self.setLayout(vbox)

        
class ScreenPlot(pg.GraphicsLayoutWidget):

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

        cmap = pg.colormap.get("CET-L9")
        bar = pg.ColorBarItem(cmap=cmap, width=10)
        max_level = pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ")["data"].max()
        bar.setLevels(low=0, high=max_level)
        bar.setImageItem([self.i1,], insert_in=self.p1)
    
    @qtc.pyqtSlot(np.ndarray)
    def update_agent_view(self, screen_data):
        self.i1.setImage(
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

    # def desired_ellipse_interaction(self, roi):
    #     if not self.was_desired_set_programmatically:
    #         x = self.desired_ellipse.pos().x()
    #         y = self.desired_ellipse.pos().y()
    #         w = self.desired_ellipse.size().x()
    #         h = self.desired_ellipse.size().y()

    #         sigma_x = w / 2 * 1e3
    #         sigma_y = h / 2 * 1e3
    #         mu_x = x * 1e3 + sigma_x
    #         mu_y = y * 1e3 + sigma_y

    #         self.user_moved_desired_ellipse.emit(mu_x, mu_y, sigma_x, sigma_y)
    #     self.was_desired_set_programmatically = False


def main():
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

    # app.aboutToQuit.connect(window.handle_application_exit)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
