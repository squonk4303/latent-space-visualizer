#!/usr/bin/env python3
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
)

import consts
from main_window import MainWindow


@pytest.fixture
def mainwindow(qtbot):
    """
    Initializes the main window of the application for use in following tests
    Returns class MainWindow(QMainWindow)
    """
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_window_basically(mainwindow, qtbot):
    """ Tests the main window and its properties """
    # Window exists
    assert mainwindow.windowTitle() == consts.WINDOW_TITLE
    assert mainwindow.centralWidget()


def test_qaction_to_switch_tabs(mainwindow, qtbot):
    """ Tests switching tabs with the QActions """
    assert mainwindow.tab_layout.currentIndex() == 0
    mainwindow.next_tab.trigger()
    assert mainwindow.tab_layout.currentIndex() == 1
    mainwindow.next_tab.trigger()
    assert mainwindow.tab_layout.currentIndex() == 0
    mainwindow.prev_tab.trigger()
    assert mainwindow.tab_layout.currentIndex() == 1

def test_buttons_to_switch_tabs(mainwindow, qtbot):
    """ Tests switching tabs with hardcoded buttons """
    qtbot.mouseClick(mainwindow.empty_tab_button, Qt.MouseButton.LeftButton)
    assert mainwindow.tab_layout.currentIndex() == 0
    qtbot.mouseClick(mainwindow.graph_tab_button, Qt.MouseButton.LeftButton)
    assert mainwindow.tab_layout.currentIndex() == 1


def test_tab_switch_after_selecting_file(mainwindow, qtbot):
    """
    Tests switching to next tab after selecting file
    """
    pass  # TODO


def test_window_starts_normally(mainwindow, qtbot):
    """
    Tests that window starts in normal state
    """
    pass  # TODO


def test_window_remembers_position(mainwindow, qtbot):
    """
    Tests that window remembers the pos and size the user left it in
    """
    pass  # TODO


def test_tab_switch_after_selecting_file(mainwindow, qtbot):
    # Menu bar is accessible and functional
    pass  # TODO



def test_file_select(mainwindow, qtbot):
    """
    Tests the action for selecting a file
    TODO: I gotta find how to check the file dialog without interrupting
    the test. And also what to assert to prove it's working as intended.
    """
    qtbot.mouseClick(mainwindow.file_menu, Qt.MouseButton.LeftButton)
    # mainwindow.action_to_open_file.trigger()
    # dialog = QApplication.activeWindow()
    # dialog.done(0)


def test_do_the_graph(mainwindow, qtbot):
    """
    Tests our graph widget
    Depends on whether we use pyqtgraph or matplotlib I suppose
    """
    pass  # TODO
