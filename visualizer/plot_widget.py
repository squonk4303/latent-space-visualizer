#!/usr/bin/env python3
import random

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)

# matplotlib necessarily imported after PyQt6
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from visualizer import consts


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

    def with_tsne(self, old_plottables):
        # Put all features in a list, and all labels in a list with corresponding indices
        # Python list comprehension is awesome; And the zip function; And tuple assignment
        labels, all_feats = tuple(
            zip(
                *tuple(
                    (key, element.features)
                    for key, value in old_plottables.items()
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
        # So it should be possible to loop over all the values in dict SavableData
        # And assign coords to the .tsne values there

        # @Wilhelmsen: Consider again whether old_plottables.values() always returns in the expected order
        for coord, pathandfeature in zip(
            coords, (p for obj in old_plottables.values() for p in obj)
        ):
            pathandfeature.tsne = coord

        # print("".join(f"{w.tsne}\n" for obj in old_plottables.values() for w in obj))

        # @Wilhelmsen: Move this assertion to tests
        # assert len(all_feats) == len(coords) == len(labels)

        # Map each label to a randomly-sampled color
        color_map = {
            label: color
            for label, color in zip(
                (category for category in old_plottables.keys()),
                random.sample(consts.COLORS, k=len(old_plottables)),
            )
        }

        for label, data in old_plottables.items():
            tsne = [obj.tsne for obj in data]
            x, y = [list(t) for t in zip(*tsne)]

            self.canvas.axes.scatter(x, y, label=label, c=color_map[label])
            self.canvas.axes.legend()

    def make_toolbar(self):
        """Generate a toolbar object for the matplotlib plot."""
        # @Wilhelmsen: Consider making this so it only generates a toolbar
        # if there's not already one registered to the parent. Else return
        # the one that is.
        toolbar = NavigationToolbar(self.canvas, self.parent)
        return toolbar


def surprise_plot(layer):
    """
    An independent plot that can appear from almost anywhere. Watch out!

    Really for use in development.
    """
    from math import sqrt, ceil

    num_kernels = layer.shape[1]
    fig, axs = plt.subplots(nrows=16, ncols=16, layout="constrained")

    # print("*** layer[1]:", "".join(f"*  {i}\n" for i in layer))
    print("layer", layer)

    # for ax, image in zip(axs.flat, images):
    for i, ax in enumerate(axs.flat):
        if i < num_kernels:
            ax.imshow(layer[0, i].cpu().numpy())
            ax.axis("off")

    plt.show()
