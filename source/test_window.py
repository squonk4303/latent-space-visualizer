#!/usr/bin/env python3
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
)

from consts import Const
from window import MainWindow, StackedLayoutManager


@pytest.fixture
def bot_mw(qtbot):
    """
    Tests for the QMainWindow class "MainWindow"
    """
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_window(bot_mw, qtbot):
    """
    Tests the main window and its properties
    """
    # Window exists
    assert bot_mw.windowTitle() == Const.WINDOW_TITLE
    assert bot_mw.centralWidget()

    # Window starts in normal state
    # Window remembers the pos and size the user left it in
    # Menu bar is accessible and functional


def test_file_select(bot_mw, qtbot):
    """
    Tests the action for selecting a file
    """
    # File select works
    # TODO: How to do the file dialog without interrupting the test
    qtbot.mouseClick(bot_mw.file_menu, Qt.MouseButton.LeftButton)
    # qtbot.mouseClick(bot_mw.open_file_button, Qt.MouseButton.LeftButton)

    # File select can be reached by menu bar and "mouse"
    # TODO: How the hell does one select an item in a menu?


def test_layout_manager():
    """
    Tests the layout manager for all it's worth
    """
    # Check correct initialization
    mylayout = StackedLayoutManager()
    assert len(mylayout.items) == 0
    assert mylayout.layout.currentWidget() is None
    assert mylayout.selected_item == -1

    # Check alt. initialization
    widget0 = QWidget()
    widget1 = QWidget()
    widget2 = QWidget()
    mylayout2 = StackedLayoutManager([widget0, widget1, widget2])
    assert len(mylayout2.items) == 3
    assert mylayout2.layout.currentWidget() is None
    assert mylayout2.selected_item == 2

    # Adding widgets
    mylayout.add_widget(widget0)
    mylayout.add_widget(widget1)
    mylayout.add_widget(widget2)
    assert len(mylayout.items) == 3
    assert mylayout.selected_item == 2

    # Scroll around
    mylayout.scroll_back()
    mylayout.scroll_back()
    assert mylayout.selected_item == 0
    mylayout.scroll_forth()
    assert mylayout.selected_item == 1

    # Scroll over until after end of list
    mylayout.scroll_forth()
    mylayout.scroll_forth()
    assert mylayout.selected_item == 0

    # Scroll back and roll over to top of list
    mylayout.scroll_back()
    assert mylayout.selected_item == 2

    # Scroll multiple
    mylayout.scroll_back(2)
    assert mylayout.selected_item == 0
    mylayout.scroll_forth(2)
    assert mylayout.selected_item == 2

    # --- Adding layers ---
    mylayout3 = StackedLayoutManager()
    layout0 = QHBoxLayout()
    mylayout3.add_layout(layout0)
    assert len(mylayout3.items) == 1
    assert mylayout3.selected_item == 0


def test_graph(bot_mw, qtbot):
    """
    Tests our graph widget
    """
    # Depends on whether we use pyqtgraph or matplotlib
    pass
