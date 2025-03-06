#!/usr/bin/env python3
from unittest.mock import patch
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog

import consts
from view_manager import MainWindow


# --- Fixtures and Sort-of-Fixtures ---
@pytest.fixture
def mainwindow(qtbot):
    """
    Initializes the main window of the application for use in following tests
    Returns class MainWindow(QMainWindow)
    """
    window = MainWindow()
    qtbot.addWidget(window)
    return window


mocked_trained_model_qfiledialog = patch.object(
    QFileDialog,
    "getOpenFileName",
    return_value=(consts.TRAINED_MODEL, consts.FILE_FILTERS["whatever"])
)


mocked_cancelled_qfiledialog = patch.object(
    QFileDialog,
    "getOpenFileName",
    return_value=("", consts.FILE_FILTERS["whatever"])
)


def test_window_basically(mainwindow, qtbot):
    """Test that the window is alive and exists."""
    assert mainwindow.windowTitle() == consts.WINDOW_TITLE
    assert mainwindow.centralWidget()


def test_qaction_to_switch_tabs(mainwindow, qtbot):
    """Test switching tabs with the QActions."""
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


@pytest.mark.slow
def test_tab_switch_after_selecting_file(mainwindow, qtbot):
    """
    Test switching to next tab after selecting file.
    Currently a bit slow, for it loads and evaluates a pretrained model...
    """
    with mocked_trained_model_qfiledialog:
        assert mainwindow.tab_layout.currentIndex() == 0
        mainwindow.action_to_open_file.trigger()
        assert mainwindow.tab_layout.currentIndex() == 0


def test_cancelled_file_select(mainwindow, qtbot):
    """Test that a cancelled file dialog exits gracefully."""
    with mocked_cancelled_qfiledialog:
        assert mainwindow.tab_layout.currentIndex() == 0
        mainwindow.action_to_open_file.trigger()
        assert mainwindow.tab_layout.currentIndex() == 0
