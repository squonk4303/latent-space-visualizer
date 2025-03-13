#!/usr/bin/env python3
import pytest

from view_manager import PrimaryWindow


@pytest.fixture
def primary_window(qtbot):
    window = PrimaryWindow()
    qtbot.addWidget(window)
    return window


def test_graph_exists(primary_window, qtbot):
    """."""
    assert primary_window.tab_layout.currentIndex() == 0
    plot_widget = getattr(primary_window, "plot", None)
    print(plot_widget)
    print(primary_window.plot)
    assert plot_widget


# canvas = primary_window.plot.canvas
# axes = primary_window.plot.canvas.axes
