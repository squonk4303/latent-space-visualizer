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
    Initializes the main window of the application for use in following tests
    Returns class MainWindow(QMainWindow)
    """
    window = MainWindow()
    qtbot.addWidget(window)
    return window

@pytest.fixture
def mylayout():
    """ Initializes a layout manager with 3 arbitrary widgets """
    widgets = (QWidget(), QWidget(), QWidget())
    layout = StackedLayoutManager(widgets)
    return layout


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


def test_layout_manager_initiation():
    emptylayout = StackedLayoutManager()
    assert len(emptylayout.items) == 0
    assert emptylayout.currentWidget() is None
    assert emptylayout.currentIndex() == -1


def test_alt_initialization():
    """
    Tests initialization by list argument
    The fixture "mylayout" does this too
    """
    widgets = (QWidget(), QWidget(), QWidget())
    mylayout = StackedLayoutManager(widgets)
    assert len(mylayout.items) == 3
    assert mylayout.currentWidget() is None
    assert mylayout.currentIndex() == -1

def test_tuple_initialization_then_method_adding():
    """
    Tests for whether initializing with a tuple destroys
    the capability to append more widgets by methods.
    """
    widgets = [QWidget(), QWidget(), QWidget()]
    mylayout = StackedLayoutManager(widgets)
    assert len(mylayout.items) == 3
    mylayout.addWidget(QWidget())
    #assert mylayout.count() == 4 TODO


def test_adding_widgets():
    """ Tests adding widgets to the layoutmanager individually by method """
    layout = StackedLayoutManager()
    layout.add_widget(QWidget())
    layout.add_widget(QWidget())
    layout.add_widget(QWidget())
    assert len(layout.items) == 3
    assert layout.count() == 3
    assert layout.currentIndex() == 0


# TEMP: Disabled
def _test_scrolling():
    """
    """
    widgets = (QWidget(), QWidget(), QWidget())
    mylayout = StackedLayoutManager(widgets)
    assert mylayout.count() == 3
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


def test_adding_layers():
    mylayout = StackedLayoutManager()
    layout0 = QHBoxLayout()
    mylayout.add_layout(layout0)
    assert len(mylayout.items) == 1
    assert mylayout.currentIndex() == 0


def test_do_the_graph(bot_mw, qtbot):
    """
    Tests our graph widget
    """
    # Depends on whether we use pyqtgraph or matplotlib
    pass
