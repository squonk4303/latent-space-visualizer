#!/usr/bin/env python3
from unittest.mock import patch
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog

import consts
from view_manager import PrimaryWindow


# --- Fixtures and Sort-of-Fixtures ---
@pytest.fixture
def primary_window(qtbot):
    """
    Initializes the primary window of the application for use in following tests
    Returns class PrimaryWindow(QMainWindow)
    """
    window = PrimaryWindow()
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


def test_window_basically(primary_window, qtbot):
    """Test that the window is alive and exists."""
    assert primary_window.windowTitle() == consts.WINDOW_TITLE
    assert primary_window.centralWidget()


def test_qaction_to_switch_tabs(primary_window, qtbot):
    """Test switching tabs with the QActions."""
    assert primary_window.tab_layout.currentIndex() == 0
    primary_window.next_tab.trigger()
    assert primary_window.tab_layout.currentIndex() == 1
    primary_window.next_tab.trigger()
    assert primary_window.tab_layout.currentIndex() == 0
    primary_window.prev_tab.trigger()
    assert primary_window.tab_layout.currentIndex() == 1


def test_buttons_to_switch_tabs(primary_window, qtbot):
    """ Tests switching tabs with hardcoded buttons """
    qtbot.mouseClick(primary_window.empty_tab_button, Qt.MouseButton.LeftButton)
    assert primary_window.tab_layout.currentIndex() == 0
    qtbot.mouseClick(primary_window.graph_tab_button, Qt.MouseButton.LeftButton)
    assert primary_window.tab_layout.currentIndex() == 1


@pytest.mark.slow
def test_tab_switch_after_selecting_file(primary_window, qtbot):
    """
    Test switching to next tab after selecting file.
    Currently a bit slow, for it loads and evaluates a pretrained model...
    """
    with mocked_trained_model_qfiledialog:
        assert primary_window.tab_layout.currentIndex() == 0
        primary_window.action_to_open_file.trigger()
        assert primary_window.tab_layout.currentIndex() == 0


def test_cancelled_file_select(primary_window, qtbot):
    """Test that a cancelled file dialog exits gracefully."""
    with mocked_cancelled_qfiledialog:
        assert primary_window.tab_layout.currentIndex() == 0
        primary_window.action_to_open_file.trigger()
        assert primary_window.tab_layout.currentIndex() == 0
