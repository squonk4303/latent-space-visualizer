#!/usr/bin/env python3
import pytest
from visualizer.view_manager import PrimaryWindow


@pytest.fixture
def primary_window(qtbot):
    window = PrimaryWindow()
    qtbot.addWidget(window)
    window.show()
    return window


def test_graph_visible(primary_window, qtbot):
    assert primary_window.tab_layout.currentIndex() == 0
    plot_widget = getattr(primary_window, "plot", None)
    assert plot_widget is not None

    assert not primary_window.plot.isVisible()
    primary_window.callable_goto_tab(1)()
    assert primary_window.plot.isVisible()


def test_toolbar_visible(primary_window, qtbot):
    assert not primary_window.toolbar.isVisible()
    primary_window.callable_goto_tab(1)()
    assert primary_window.toolbar.isVisible()


# canvas = primary_window.plot.canvas
# axes = primary_window.plot.canvas.axes
