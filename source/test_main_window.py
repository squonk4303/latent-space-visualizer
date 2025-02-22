#!/usr/bin/env python3
import pytest

from PyQt6.QtCore import Qt

import consts
from main_window import MainWindow


@pytest.fixture
def bot_mw(qtbot):
    """
    Initializes the main window of the application for use in following tests
    Returns class MainWindow(QMainWindow)
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

    # --- TODO: Move to layout manager block ---
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


def test_do_the_graph(bot_mw, qtbot):
    """
    Tests our graph widget
    """
    # Depends on whether we use pyqtgraph or matplotlib
    pass
