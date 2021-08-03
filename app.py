import sys
from time import sleep

import matplotlib
from numpy.core.numeric import extend_all
from scipy.ndimage import interpolation
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import pydoocs

# from applogic import Agent


class ScreenView(FigureCanvasQTAgg):

    def __init__(self, resolution, pixel_size):
        self.resolution = resolution
        self.pixel_size = pixel_size

        self.fig = Figure()
        self.ax_live = self.fig.add_subplot(121)
        self.ax_agent = self.fig.add_subplot(122)

        super().__init__(self.fig)

        self.screen_extent = (-self.resolution[0] * self.pixel_size[0] / 2 * 1e3,
                              self.resolution[0] * self.pixel_size[0] / 2 * 1e3,
                              -self.resolution[1] * self.pixel_size[1] / 2 * 1e3,
                              self.resolution[1] * self.pixel_size[1] / 2 * 1e3)
        self.create_plot()

        self.setFixedSize(1300, 400)
    
    def create_plot(self):
        self.ax_live.set_title("Screen View (Live)")
        self.live_screen_plot = self.ax_live.imshow(
            np.zeros(self.resolution), 
            cmap="magma",
            interpolation="None",
            extent=self.screen_extent
        )
        self.ax_live.set_xlabel("x (mm)")
        self.ax_live.set_ylabel("y (mm)")

        self.ax_agent.set_title("Screen View (Agent)")
        self.agent_screen_plot = self.ax_agent.imshow(
            np.zeros(self.resolution), 
            cmap="magma",
            interpolation="None",
            extent=self.screen_extent
        )
        self.ax_agent.set_xlabel("x (mm)")
        self.ax_agent.set_ylabel("y (mm)")

        self.fig.tight_layout()
    
    @qtc.pyqtSlot(np.ndarray)
    def update_screen_live(self, screen_data):
        self.live_screen_plot.set_data(screen_data)
        self.live_screen_plot.set_clim(vmin=0, vmax=screen_data.max())

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

    def __init__(self):
        super().__init__()

        self.agent = Agent()

    def run(self):
        while True:
            self.spectralvd.read_crisp()
            
            ann_current = self.spectralvd.ann_reconstruction()
            self.ann_current_updated.emit(ann_current)

            nils_current = self.spectralvd.nils_reconstruction()
            self.nils_current_updated.emit(nils_current)

            sleep(0.1)
    
    def change_grating(self, grating):
        print(f"Changing grating to {grating}")


class App(qtw.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Autonomous Beam Position and Focusing")

        self.raw_screen_view = ScreenView((2448,2040), (3.3198e-6,2.4469e-6))
        
        self.read_thread = AcceleratorReadThread()
        self.read_thread.screen_updated.connect(self.raw_screen_view.update_screen_live)

        # self.interface_thread = AcceleratorInterfaceThread()
        # self.interface_thread.ann_current_updated.connect(self.current_plot.update_ann)
        # self.interface_thread.nils_current_updated.connect(self.current_plot.update_nils)

        self.set_position_heading = qtw.QLabel("Choose Position")
        self.mu_x_slider = qtw.QSlider(qtc.Qt.Vertical)
        self.mu_x_slider.setMinimum(0)
        self.mu_x_slider.setMaximum(11)
        # self.mu_x_slider.valueChanged.connect(self.set_shutter_speed)

        self.grating_dropdown = qtw.QComboBox()
        self.grating_dropdown.addItems(["low", "high", "both"])
        # self.grating_dropdown.currentTextChanged.connect(self.interface_thread.change_grating)

        hbox = qtw.QHBoxLayout()
        hbox.addWidget(self.raw_screen_view)
        vbox = qtw.QVBoxLayout()
        vbox.addWidget(self.set_position_heading)
        vbox.addWidget(self.mu_x_slider)
        vbox.addStretch()
        hbox.addLayout(vbox)
        self.setLayout(hbox)

        self.read_thread.start()
    
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
