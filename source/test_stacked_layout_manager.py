#!/usr/bin/env python3
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
)

import consts
from stacked_layout_manager import StackedLayoutManager


@pytest.fixture
def mylayout():
    """ Initializes a layout manager with 3 arbitrary widgets """
    widgets = (QWidget(), QWidget(), QWidget())
    layout = StackedLayoutManager(widgets)
    return layout


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
    # assert mylayout.count() == 4 TODO


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
