#!/usr/bin/env python3
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
)

import consts
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
    assert bot_mw.windowTitle() == consts.WINDOW_TITLE
    assert bot_mw.centralWidget()

    # Different ways to switch to other tabs
    # With TEMP button
    assert bot_mw.tab_layout.currentIndex() == 0
    qtbot.mouseClick(bot_mw.TEMP_button, Qt.MouseButton.LeftButton)
    assert bot_mw.TEMP_label.text() == "Hi welcome to the graph tab :3"
    assert bot_mw.tab_layout.currentIndex() == 1
    qtbot.mouseClick(bot_mw.empty_tab_button, Qt.MouseButton.LeftButton)

    # With the QActions
    assert bot_mw.tab_layout.currentIndex() == 0
    bot_mw.next_tab.trigger()
    assert bot_mw.tab_layout.currentIndex() == 1
    bot_mw.next_tab.trigger()
    assert bot_mw.tab_layout.currentIndex() == 0
    bot_mw.prev_tab.trigger()
    assert bot_mw.tab_layout.currentIndex() == 1

    # With the tab buttons
    qtbot.mouseClick(bot_mw.empty_tab_button, Qt.MouseButton.LeftButton)
    assert bot_mw.tab_layout.currentIndex() == 0
    qtbot.mouseClick(bot_mw.graph_tab_button, Qt.MouseButton.LeftButton)
    assert bot_mw.tab_layout.currentIndex() == 1

    # --- TODO ---
    # Switches to next tab after selecting file

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
    # bot_mw.action_to_open_file.trigger()
    # dialog = QApplication.activeWindow()
    # dialog.done(0)

    # File select can be reached by menu bar and "mouse"
    # TODO: How the hell does one select an item in a menu?


def test_layout_manager():
    """
    Tests the layout manager for all it's worth
    Doesn't even use the qtbot. Just tests the class.
    """
    # Check correct initialization
    mylayout = StackedLayoutManager()
    assert len(mylayout.items) == 0
    assert mylayout.currentWidget() is None
    assert mylayout.currentIndex() == -1

    # Check alt. initialization
    widget0 = QWidget()
    widget1 = QWidget()
    widget2 = QWidget()
    mylayout2 = StackedLayoutManager([widget0, widget1, widget2])
    assert len(mylayout2.items) == 3
    assert mylayout2.currentWidget() is None
    assert mylayout2.currentIndex() == -1

    # Adding widgets
    mylayout.add_widget(widget0)
    mylayout.add_widget(widget1)
    mylayout.add_widget(widget2)
    assert len(mylayout.items) == 3
    assert mylayout.currentIndex() == 0

    # Scroll around
    mylayout.scroll_back()
    mylayout.scroll_back()
    assert mylayout.currentIndex() == 1
    mylayout.scroll_forth()
    assert mylayout.currentIndex() == 2

    # Scroll over until after end of list
    mylayout.scroll_forth()
    mylayout.scroll_forth()
    assert mylayout.currentIndex() == 1

    # Scroll back and roll over to top of list
    mylayout.scroll_back()
    assert mylayout.currentIndex() == 0

    # Scroll multiple
    mylayout.scroll_somewhere(-2)
    assert mylayout.currentIndex() == 1
    mylayout.scroll_somewhere(2)
    assert mylayout.currentIndex() == 0

    # --- Adding layers ---
    mylayout3 = StackedLayoutManager()
    layout0 = QHBoxLayout()
    mylayout3.add_layout(layout0)
    assert len(mylayout3.items) == 1
    assert mylayout3.currentIndex() == 0


def test_graph(bot_mw, qtbot):
    """
    Tests our graph widget
    """
    # Depends on whether we use pyqtgraph or matplotlib
    pass
