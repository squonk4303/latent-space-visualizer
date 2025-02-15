#!/usr/bin/env python3
import pytest
from PyQt6.QtCore import Qt
from window import SomeWindow, MainWindow, Const


"""
Tests for the QMainWindow class "MainWindow"
"""
@pytest.fixture
def bot_mw(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_window(bot_mw, qtbot):
    """ Tests the main window and its properties """
    # It exists
    assert bot_mw.windowTitle() == Const["WINDOW_TITLE"]
    assert bot_mw.centralWidget()
    # It starts in normal state
    # There's a menu bar
    # It remembers the pos and size the user left it in
    pass


def test_file_select(bot_mw, qtbot):
    """ Tests the action for selecting a file """
    pass


def test_graph(bot_mw, qtbot):
    """ Tests our graph widget """
    # Depends on whether we use pyqtgraph or matplotlib
    pass


"""
Tests for the Widget class "SomeWindow"
"""
@pytest.fixture
def app(qtbot):
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
