#!/usr/env/bin python3

import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)

# matplotlib necessarily imported after PyQt6
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# TODO: Set matplotlib backend explicitly somewhere
# https://matplotlib.org/stable/users/explain/figure/backends.html


class MplCanvas(FigureCanvasQTAgg):
    """Make a canvas for the plot to render onto."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(1, 1, 1)
        super().__init__(fig)


class PlotWidget(QWidget):
    """Define and draw a graphical plot."""
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self)
        self.plot_sine()
        layout.addWidget(self.canvas)

    def plot_sine(self):
        x = np.linspace(0, 20, 50)
        y = np.sin(x)
        self.canvas.axes.plot(x, y)
        self.canvas.draw()
