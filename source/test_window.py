#!/usr/bin/env python3
import pytest
from PyQt6.QtCore import Qt
from window import SomeWindow, MainWindow, Const


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
    assert bot_mw.windowTitle() == Const["WINDOW_TITLE"]
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


def test_graph(bot_mw, qtbot):
    """
    Tests our graph widget
    """
    # Depends on whether we use pyqtgraph or matplotlib
    pass


@pytest.fixture
def app(qtbot):
    """
    Tests for the Widget class "SomeWindow"
    """
    window = SomeWindow()
    qtbot.addWidget(window)
    return window


def test_typing(app, qtbot):
    assert app.lineEdit.text() == ""

    qtbot.keyClicks(app.lineEdit, "Bogus")
    assert app.lineEdit.text() == "Bogus"


def test_clearing(app, qtbot):
    qtbot.keyClicks(app.lineEdit, "Bogus")
    assert app.lineEdit.text() == "Bogus"

    qtbot.mouseClick(app.button, Qt.MouseButton.LeftButton)
    assert app.lineEdit.text() == ""
