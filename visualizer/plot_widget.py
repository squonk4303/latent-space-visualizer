#!/usr/bin/env python3
from contextlib import nullcontext

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)
from PyQt6.QtGui import QPalette

# matplotlib necessarily imported after PyQt6
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from pathlib import Path
import matplotlib.pyplot as plt
import PIL

from visualizer import consts


class MplCanvas(FigureCanvasQTAgg):
    """Hold a canvas for the plot to render onto."""

    def __init__(self, parent, width=5, height=4, dpi=100, background="0.85", forecolor="1"):
        fig = Figure(figsize=(width, height), dpi=dpi, layout="constrained", facecolor=background)
        # Setting Foreground Colors
        plt.rcParams['text.color'] = forecolor            # All text (titles, annotations)
        plt.rcParams['axes.labelcolor'] = forecolor       # Axis labels
        plt.rcParams['xtick.color'] = forecolor           # X tick labels
        plt.rcParams['ytick.color'] = forecolor           # Y tick labels
        # contextlib.nullcontext being a context manager which does nothing
        cm = plt.xkcd() if consts.flags["xkcd"] else nullcontext()
        with cm:
            self.input_display, self.scatterplot, self.output_display = fig.subplots(
                nrows=1, ncols=3
            )
        
        self.input_display.get_xaxis().set_visible(False)
        self.input_display.get_yaxis().set_visible(False)
        self.output_display.get_xaxis().set_visible(False)
        self.output_display.get_yaxis().set_visible(False)

        super().__init__(fig)

    def redraw(self, in_imgdesc="", out_imgdesc=""):
        """Clear subplots and reapply titles."""
        self.scatterplot.clear()
        self.input_display.clear()
        self.output_display.clear()

        self.scatterplot.set_title("Visualized Latent Space")
        self.scatterplot.set_xlabel("X = Dimesion 1")
        self.scatterplot.set_ylabel("Y = Dimension 2")
        self.input_display.set_title("Input Image")
        self.output_display.set_title("Output Image")
        self.input_display.text(
            0.5, -0.01, in_imgdesc, ha="center", va="top", 
            transform=self.input_display.transAxes
            )
        self.output_display.text(
            0.5, -0.01, "Dominant Category: "+out_imgdesc.capitalize(), ha="center", va="top", 
            transform=self.output_display.transAxes
            )
        

class PlotWidget(QWidget):
    def __init__(self, parent):
        """Define and draw a graphical plot."""
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout(self)
        self.bgcolor = self.get_color()
        self.fgcolor = self.get_color(consts.COLOR.TEXT)
        self.canvas = MplCanvas(self, background=self.bgcolor, forecolor=self.fgcolor)
        self.canvas.redraw()
        self.canvas.draw()
        self.canvas.flush_events()
        layout.addWidget(self.canvas)

    def get_color(self, color=consts.COLOR.BACKGROUND):
        match color:
            case consts.COLOR.BACKGROUND:
                background_color = self.palette().color(QPalette.ColorRole.Window)
                return background_color.name()
            case consts.COLOR.TEXT:
                text_color = self.palette().color(QPalette.ColorRole.WindowText)
                return text_color.name()


    def the_plottables(self, labels, paths, coords, masks, colormap):
        # @Wilhelmsen: Make it detect whether coords are 2d or 3d and act accordingly
        # Map each label to a randomly-sampled color
        unique_labels = list(set(labels))

        # Make a dict which maps paths and coords to related unique labels
        plottables = {key: {"paths": [], "coords": []} for key in unique_labels}


        for L, p, c in zip(labels, paths, coords):
            plottables[L]["paths"].append(p)
            plottables[L]["coords"].append(c)

        # Clear subplot before plotting
        self.canvas.scatterplot.clear()
        for L in sorted(plottables.keys()):
            x, y = zip(*plottables[L]["coords"])
            self.canvas.scatterplot.scatter(x, y, label=L.capitalize(), c=colormap[L])

        # Styling
        # @Linnea: Move this to MplCanvas
        self.canvas.scatterplot.set_facecolor('1')
        self.canvas.scatterplot.axvline(x=0, linestyle='--', linewidth=0.4, color='0.4')
        self.canvas.scatterplot.axhline(y=0, linestyle='--', linewidth=0.4, color='0.4')
        self.canvas.scatterplot.set_xlim(-2,2)
        self.canvas.scatterplot.set_ylim(-2,2)
        self.canvas.scatterplot.legend(loc="upper left", bbox_to_anchor=(1,1), framealpha=0)

        self.canvas.draw()
        self.canvas.flush_events()

    def new_tuple(self, value, labels, paths, coords, masks, colormap):
        """
        Changes which input image and mask is displayed, and highlight the corresponding point.

        Made to be called when user selects a given data point, which shows
        the related image, mask, and highlights the corresponding point.
        """
        filename = Path(paths[value]).name
        inpic = PIL.Image.open(paths[value])
        self.canvas.redraw(filename,labels[value])
        tx, ty = coords[value]
        self.the_plottables(labels, paths, coords, masks, colormap)
        self.canvas.input_display.imshow(inpic)
        self.canvas.output_display.imshow(masks[value])
        self.canvas.scatterplot.scatter(tx, ty, s=500, marker="+", c="black")
        # Update functionality to display correctly 
        self.canvas.draw()
        self.canvas.flush_events()

    def make_toolbar(self):
        """Generate a toolbar object for the matplotlib plot."""
        # @Wilhelmsen: Consider making this so it only generates a toolbar
        # if there's not already one registered to the parent. Else return
        # the one that is.
        toolbar = NavigationToolbar(self.canvas, self.parent)
        return toolbar
