#!/usr/env/bin python3
import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)

# matplotlib necessarily imported after PyQt6
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from visualizer import consts, parse


class MplCanvas(FigureCanvasQTAgg):
    """Hold a canvas for the plot to render onto."""

    def __init__(self, parent, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        if consts.flags["xkcd"]:
            with plt.xkcd():
                self.axes = fig.add_subplot(1, 1, 1)
        else:
            self.axes = fig.add_subplot(1, 1, 1)

        super().__init__(fig)


class PlotWidget(QWidget):
    def __init__(self, parent):
        """Define and draw a graphical plot."""
        super().__init__(parent)

        self.parent = parent
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self)

        layout.addWidget(self.canvas)

    def plot_sine(self):
        x = np.linspace(0, 20, 50)
        y = np.sin(x)
        self.canvas.axes.plot(x, y)
        self.canvas.draw()

    def plot_from_csv(self, filepath):
        data = parse.csv_as_list(filepath)
        self.canvas.axes.plot(data)
        self.canvas.draw()

    def plot_from_2d(self, array_2d: np.ndarray):
        """
        Plot a scatterplot from the given array.

        Does not clear previously plotted data.
        """
        x = array_2d[:, 0]
        y = array_2d[:, 1]

        if array_2d.shape[1] == 2:
            self.canvas.axes.scatter(x, y)
        elif array_2d.shape[1] == 3:
            z = array_2d[:, 2]
            self.canvas.axes.scatter(x, y, z)

        self.canvas.draw()

    def make_toolbar(self):
        """Generate a toolbar object for the matplotlib plot."""
        toolbar = NavigationToolbar(self.canvas, self.parent)
        return toolbar
