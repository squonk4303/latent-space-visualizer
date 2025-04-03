#!/usr/env/bin python3
import numpy as np
import random
from sklearn.manifold import TSNE

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

    def with_tsne(self, plottables):
        # Get random sample of mpl-compliant colors
        colors = random.sample(consts.COLORS, len(consts.COLORS))

        # labels = [k for k in plottables.keys()]
        # print("".join(f"{key}, {element}\n" for key, value in dict.items() for element in value])

        """
        labels = [key, element for key, value in plottable.items() for elemene in value]
        """

        # all_feats = [t.features for v in plottables.values() for t in v]

        labels, all_feats = tuple(
            zip(
                *list(
                    (key, element.features)
                    for key, value in plottables.items()
                    for element in value
                )
            )
        )

        print("--- labels:", labels)
        print("--- all_feats:", all_feats)
        print("".join(f"{t.shape}\t" for t in all_feats))
        all_feats = np.array(all_feats).reshape(len(all_feats), -1)
        print("".join(f"{t.shape}\t" for t in all_feats))

        # t-SNE the features
        perplexity_value = min(30, len(all_feats) - 1)
        tsne_conf = TSNE(
            n_components=2,
            perplexity=perplexity_value,
            random_state=consts.seed,
        )

        coords = tsne_conf.fit_transform(all_feats)

        print("--- t-SNE ---")
        print("\n".join([f"{t[0]}\t{t[1]}" for t in coords]))

        for i, j in zip(labels, coords):
            # self.canvas.axes.scatter(
            pass

    def make_toolbar(self):
        """Generate a toolbar object for the matplotlib plot."""
        # @Wilhelmsen: Consider making this so it only generates a toolbar
        # if there's not already one registered to the parent. Else return
        # the one that is.
        toolbar = NavigationToolbar(self.canvas, self.parent)
        return toolbar
