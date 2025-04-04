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
        # Put all features in a list, and all labels in a list with corresponding indices
        # Python list comprehension is awesome; And the zip function; And tuple assignment
        labels, all_feats = tuple(
            zip(
                *tuple(
                    (key, element.features)
                    for key, value in plottables.items()
                    for element in value
                )
            )
        )

        # Make sure the feats are represented as a numpy.ndarray
        all_feats = np.array(all_feats).reshape(len(all_feats), -1)

        # t-SNE the features
        perplexity_value = min(30, len(all_feats) - 1)
        tsne_conf = TSNE(
            n_components=2,
            perplexity=perplexity_value,
            random_state=consts.seed,
        )
        coords = tsne_conf.fit_transform(all_feats)
        # print("".join([f"{x}\t{y}\n" for x, y in coords]))

        # Coords is now an iter
        # So it should be possible to loop over all the values in dict Plottables
        # And assign coords to the .tsne values there

        # @Wilhelmsen: Consider again whether plottables.values() always returns in the expected order
        for coord, pathandfeature in zip(
            coords, [p for obj in plottables.values() for p in obj]
        ):
            pathandfeature.tsne = coord

        print("".join(f"{w.tsne}\n" for obj in plottables.values() for w in obj))

        # @Wilhelmsen: Move this assertion to tests
        # assert len(all_feats) == len(coords) == len(labels)

        # Map each label to a color
        color_map = {
            label: color
            for label, color in zip(
                [k for k in plottables.keys()],
                random.sample(consts.COLORS, k=len(plottables)),
            )
        }

        for label, data in plottables.items():
            tsne = [obj.tsne for obj in data]
            # print(label, tsne)
            print("".join(f"{label}, {i}\n" for i in tsne))
            x, y = [list(t) for t in zip(*tsne)]
            # print("".join(f"{label}, {i}\n" for i in transformed))

            self.canvas.axes.scatter(x, y, label=label, c=color_map[label])
            self.canvas.axes.legend()

    def make_toolbar(self):
        """Generate a toolbar object for the matplotlib plot."""
        # @Wilhelmsen: Consider making this so it only generates a toolbar
        # if there's not already one registered to the parent. Else return
        # the one that is.
        toolbar = NavigationToolbar(self.canvas, self.parent)
        return toolbar
