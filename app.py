import sys

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qt_material import apply_stylesheet

from ea_optimize import optimize_async


class RLAgentEAController:
    """Controller to connect `RLAgentEAWidget` to `ea_optimize.optimze`."""

    def __init__(self, model, view):
        self.optimize = model
        self.view = view
        self.connect_signals_to_slots()
        self.setup_initial_state()

    def connect_signals_to_slots(self):
        """
        Connect signals and slots of view and controller to enable view to controll the
        model.
        """
        # Connect target line edits to target ellipse
        for line_edit in self.view.target_line_edits.values():
            line_edit.editingFinished.connect(self.update_target)

        # Connect max steps and threshold checkboxes to their line edit enabledness
        self.view.max_steps_checkbox.stateChanged.connect(
            self.view.max_steps_line_edit.setEnabled
        )
        self.view.threshold_checkbox.stateChanged.connect(
            self.view.threshold_line_edit.setEnabled
        )

        # Connect optimise button to starting the optimisation function
        self.view.start_stop_button.clicked.connect(self.start_optimization)

    def setup_initial_state(self):
        """Put app into state expected at start-up."""
        self.view.max_steps_checkbox.setChecked(True)
        self.view.threshold_checkbox.setChecked(False)
        self.view.threshold_line_edit.setEnabled(
            False
        )  # Hacky: I would prefer this to happen via the signal

    def update_target(self):
        """Is called every time that the target beam parameters are changed."""
        target = {
            k: 1e-3 * float(self.view.target_line_edits[k].text())
            for k in ["mu_x", "sigma_x", "mu_y", "sigma_y"]
        }  # Convert from mm to m
        self.view.set_target_entries(**target)
        self.view.place_target_ellipse(**target)

    def start_optimization(self):
        """
        Get target and other configurations from the GUI and initiate the optimisation.
        """
        optimize_async(
            target_mu_x=float(self.view.target_line_edits["mu_x"].text()) * 1e3,
            target_sigma_x=float(self.view.target_line_edits["sigma_x"].text()) * 1e3,
            target_mu_y=float(self.view.target_line_edits["mu_y"].text()) * 1e3,
            target_sigma_y=float(self.view.target_line_edits["sigma_y"].text()) * 1e3,
            target_mu_x_threshold=float(self.view.threshold_line_edit.text()) * 1e3
            if self.view.threshold_checkbox.isChecked()
            else 3.3198e-6,
            target_mu_y_threshold=float(self.view.threshold_line_edit.text()) * 1e3
            if self.view.threshold_checkbox.isChecked()
            else 3.3198e-6,
            target_sigma_x_threshold=float(self.view.threshold_line_edit.text()) * 1e3
            if self.view.threshold_checkbox.isChecked()
            else 3.3198e-6,
            target_sigma_y_threshold=float(self.view.threshold_line_edit.text()) * 1e3
            if self.view.threshold_checkbox.isChecked()
            else 3.3198e-6,
            max_steps=int(self.view.max_steps_line_edit.text())
            if self.view.max_steps_checkbox.isChecked()
            else None,
            model_name="chocolate-totem-247",
            logbook=True,
            callback=None,
        )


class RLAgentEAWidget(QWidget):
    """Widget for controlling EA RL agnet."""

    def __init__(self):
        super().__init__()

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.create_optimisation_configuration()
        self.create_screen_view()

    def create_optimisation_configuration(self):
        """Make interface for configuring optimisation."""
        target_form_layout = QFormLayout()
        self.main_layout.addLayout(target_form_layout)

        self.target_line_edits = {
            k: QLineEdit("0.0") for k in ["mu_x", "sigma_x", "mu_y", "sigma_y"]
        }
        for name in ["mu_x", "sigma_x", "mu_y", "sigma_y"]:
            pretty_name = name.replace("mu_", "μ").replace("sigma_", "σ")
            target_form_layout.addRow(
                f"{pretty_name} (mm)", self.target_line_edits[name]
            )
        for name in ["mu_x", "mu_y"]:
            self.target_line_edits[name].setValidator(QDoubleValidator(-3, 3, -1))
        for name in ["sigma_x", "sigma_y"]:
            self.target_line_edits[name].setValidator(QDoubleValidator(0, 3, -1))

        self.max_steps_checkbox = QCheckBox("Max steps")
        self.max_steps_line_edit = QLineEdit("25")
        self.max_steps_line_edit.setValidator(QIntValidator(0, 300))
        target_form_layout.addRow(self.max_steps_checkbox, self.max_steps_line_edit)

        self.threshold_checkbox = QCheckBox("Threshold (mm)")
        self.threshold_line_edit = QLineEdit("0.01")
        self.threshold_line_edit.setValidator(QDoubleValidator(0, 3.0, -1))
        target_form_layout.addRow(self.threshold_checkbox, self.threshold_line_edit)

        self.start_stop_button = QPushButton("Optimise")
        target_form_layout.addRow(self.start_stop_button)

    def create_screen_view(self):
        """Make view of screen."""
        resolution = (2448, 2040)
        pixel_size = (3.3198e-6, 2.4469e-6)
        self.screen_rect = (
            -resolution[0] * pixel_size[0] / 2 * 1e3,
            -resolution[1] * pixel_size[1] / 2 * 1e3,
            resolution[0] * pixel_size[0] * 1e3,
            resolution[1] * pixel_size[1] * 1e3,
        )

        plot_widget = pg.PlotWidget()
        plot_widget.setMouseEnabled(x=False, y=False)
        plot_widget.setRange(
            xRange=(self.screen_rect[0], self.screen_rect[0] + self.screen_rect[2]),
            yRange=(self.screen_rect[1], self.screen_rect[1] + self.screen_rect[3]),
            padding=0,
        )
        plot_widget.getAxis("bottom").setLabel("x (mm)")
        plot_widget.getAxis("left").setLabel("y (mm)")
        plot_widget.hideButtons()
        plot_widget.setAspectLocked()
        self.main_layout.addWidget(plot_widget)

        self.image_item = pg.ImageItem()
        plot_widget.addItem(self.image_item)
        self.show_screen_image(np.zeros(list(reversed(resolution))))

        self.target_ellipse_item = pg.EllipseROI(
            (0, 0),
            (1e-10, 1e-10),  # Not zero to avoid division by zero
            pen=pg.mkPen("w", width=2),
            rotatable=False,
            movable=False,
            resizable=False,
        )
        for handle in self.target_ellipse_item.getHandles():
            self.target_ellipse_item.removeHandle(handle)
        plot_widget.addItem(self.target_ellipse_item)

        self.current_ellipse_item = pg.EllipseROI(
            (0, 0),
            (1e-10, 1e-10),  # Not zero to avoid division by zero
            pen=pg.mkPen("w", width=2),
            rotatable=False,
            movable=False,
            resizable=False,
        )
        for handle in self.current_ellipse_item.getHandles():
            self.current_ellipse_item.removeHandle(handle)
        plot_widget.addItem(self.current_ellipse_item)

        cmap = pg.colormap.get("CET-L9")
        bar = pg.ColorBarItem(colorMap=cmap, width=10)
        bar.setLevels(low=0, high=2**12)
        bar.setImageItem(self.image_item)  # , insert_in=plot_widget)

    def show_screen_image(self, img):
        """Set the image that is show in the screen plot."""
        self.image_item.setImage(
            np.flipud(img),
            axisOrder="row-major",
            rect=self.screen_rect,
            autoLevels=False,
        )

    def place_current_ellipse(self, mu_x, sigma_x, mu_y, sigma_y):
        """Move the current ellipse to the given beam parameters."""
        mu_x, sigma_x, mu_y, sigma_y = [
            x * 1e3 for x in (mu_x, sigma_x, mu_y, sigma_y)
        ]  # Convert to mm

        self.current_ellipse_item.setPos((mu_x - sigma_x, mu_y - sigma_y))
        self.current_ellipse_item.setSize((2 * sigma_x, 2 * sigma_y))

    def place_target_ellipse(self, mu_x, sigma_x, mu_y, sigma_y):
        """Move the target ellipse to the given beam parameters."""
        mu_x, sigma_x, mu_y, sigma_y = [
            x * 1e3 for x in (mu_x, sigma_x, mu_y, sigma_y)
        ]  # Convert to mm
        sigma_x, sigma_y = [
            max(x, 1e-10) for x in [sigma_x, sigma_y]
        ]  # Prevent division by zero

        self.target_ellipse_item.setPos((mu_x - sigma_x, mu_y - sigma_y))
        self.target_ellipse_item.setSize((2 * sigma_x, 2 * sigma_y))

    def set_target_entries(self, mu_x, sigma_x, mu_y, sigma_y):
        """
        Enter the given beam parameters into the line edits for the target beam
        parameters.
        """
        mu_x, sigma_x, mu_y, sigma_y = [
            x * 1e3 for x in (mu_x, sigma_x, mu_y, sigma_y)
        ]  # Convert to mm

        self.target_line_edits["mu_x"].setText(str(mu_x))
        self.target_line_edits["sigma_x"].setText(str(sigma_x))
        self.target_line_edits["mu_y"].setText(str(mu_y))
        self.target_line_edits["sigma_y"].setText(str(sigma_y))


class RLAgentEAWindow(QMainWindow):
    """
    Main window of the RL agent app.

    Note: This is currently seperate from `RLAgentEAWidget` becuase we might add
    further agents later.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Agent EA")
        rl_agent_ea_widget = RLAgentEAWidget()
        self.setCentralWidget(rl_agent_ea_widget)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Force the style to be the same on all OSs
    apply_stylesheet(app, theme="dark_amber.xml")

    window = RLAgentEAWindow()
    window.show()

    controller = RLAgentEAController(model=None, view=window.centralWidget())

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
