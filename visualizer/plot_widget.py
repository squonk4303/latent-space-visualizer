#!/usr/bin/env python3
from contextlib import nullcontext
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
import PIL

from visualizer import consts


class MplCanvas(FigureCanvasQTAgg):
    """Hold a canvas for the plot to render onto."""

    def __init__(self, parent, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        # contextlib.nullcontext being a context manager which does nothing
        cm = plt.xkcd() if consts.flags["xkcd"] else nullcontext()
        with cm:
            self.input_display, self.axes, self.output_display = fig.subplots(
                nrows=1, ncols=3
            )

        super().__init__(fig)


class PlotWidget(QWidget):
    def __init__(self, parent):
        """Define and draw a graphical plot."""
        super().__init__(parent)

        self.parent = parent
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self)
        self.canvas.input_display.set_title("Input Image")
        self.canvas.output_display.set_title("Output Mask")
        self.canvas.axes.set_title("Visualized Latent Space")
        self.canvas.axes.set_xlabel("X = placeholder")
        self.canvas.axes.set_ylabel("Y = placeholder")

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

    def the_plottables(self, labels, paths, coords, masks):
        # @Wilhelmsen: Make it detect whether coords are 2d or 3d and act accordingly
        # Map each label to a randomly-sampled color
        unique_labels = list(set(labels))
        color_map = {
            label: color
            for label, color in zip(
                unique_labels,
                random.sample(consts.COLORS, k=len(unique_labels)),
            )
        }

        # Make a dict which maps paths and coords to related unique labels
        plottables = {key: {"paths": [], "coords": []} for key in unique_labels}

        for L, p, c in zip(labels, paths, coords):
            plottables[L]["paths"].append(p)
            plottables[L]["coords"].append(c)

        for L in plottables:
            x, y = zip(*plottables[L]["coords"])
            self.canvas.axes.scatter(x, y, label=L, c=color_map[L])

        self.canvas.axes.legend(loc="best")

        NUMBER = 12
        inpic = PIL.Image.open(paths[NUMBER])
        self.canvas.input_display.imshow(inpic)
        self.canvas.output_display.imshow(masks[NUMBER])

        tx, ty = coords[NUMBER][0], coords[NUMBER][1]
        print("*** tx, ty:", tx, ty)
        self.canvas.axes.scatter(
            tx, ty, s=1000, marker="+", c="red"
        )

    def with_tsne(self, old_plottables):
        """Sucks and is bad."""
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
