import sys

import numpy as np
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
import pyqtgraph as pg


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
            k: QLineEdit("0.00") for k in ["μx", "σx", "μy", "σy"]
        }
        for name in ["μx", "σx", "μy", "σy"]:
            target_form_layout.addRow(f"{name} (mm)", self.target_line_edits[name])

        max_steps_checkbox = QCheckBox("Max steps")
        max_steps_line_edit = QLineEdit("25")
        target_form_layout.addRow(max_steps_checkbox, max_steps_line_edit)

        threshold_checkbox = QCheckBox("Threshold (mm)")
        threshold_line_edit = QLineEdit("0.01")
        target_form_layout.addRow(threshold_checkbox, threshold_line_edit)

        start_stop_button = QPushButton("Optimise")
        target_form_layout.addRow(start_stop_button)

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
            (0, 0),
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
            (0, 0),
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

        self.target_ellipse_item.setPos((mu_x - sigma_x, mu_y - sigma_y))
        self.target_ellipse_item.setSize((2 * sigma_x, 2 * sigma_y))


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

    window = RLAgentEAWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
