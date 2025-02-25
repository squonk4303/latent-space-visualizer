#!/usr/bin/env python3
import pytest

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLayout,
    QVBoxLayout,
    QWidget,
)

from stacked_layout_manager import StackedLayoutManager


@pytest.fixture
def filledlayout():
    """ Initializes a layout manager with 3 arbitrary widgets """
    widgets = (QWidget(), QWidget(), QWidget())
    layout = StackedLayoutManager(widgets)
    return layout


def test_layout_manager_initiation():
    """ Tests initialization of a layout """
    emptylayout = StackedLayoutManager()
    assert emptylayout.count() == 0
    assert emptylayout.currentWidget() is None
    assert emptylayout.currentIndex() == -1


def test_alt_initialization():
    """ Tests initialization by list argument """
    widgets = (QWidget(), QWidget(), QWidget())
    layout = StackedLayoutManager(widgets)
    assert layout.count() == 3
    assert layout.currentWidget() == widgets[0]
    assert layout.currentIndex() == 0


def test_adding_layers():
    """
    Tests adding QHBoxLayouts by method. QHBoxlayouts are just examples of
    layouts. This should be valid for all layouts except QLayouts.
    See below error test.
    """
    layout = StackedLayoutManager()
    layouts = (QHBoxLayout(), QVBoxLayout(), QHBoxLayout())
    layout.add_layout(layouts[0])
    layout.add_layout(layouts[1])
    layout.add_layout(layouts[2])
    assert layout.count() == 3
    assert layout.currentIndex() == 0


def test_alt_initalization_with_qlayouts_error():
    """
    Tries to initiate QLayouts by list argument,
    and fails at putting QLayouts in a list.
    NOTE: This fails because QLayouts are a C++ abstract class
    """
    with pytest.raises(TypeError):
        layouts = (QLayout(), QLayout(), QLayout())
        layout = StackedLayoutManager(layouts)
        assert layout.count() == 3
        assert layout.currentlayout() == layouts[0]
        assert layout.currentIndex() == 0


def test_tuple_initialization_then_method_adding(filledlayout):
    """
    Tests for whether initializing with a tuple destroys
    the capability to append more widgets by method
    """
    assert filledlayout.count() == 3
    filledlayout.addWidget(QWidget())
    assert filledlayout.count() == 4


def test_adding_widgets():
    """ Tests adding widgets to the layoutmanager individually by method """
    layout = StackedLayoutManager()
    layout.add_widget(QWidget())
    layout.add_widget(QWidget())
    layout.add_widget(QWidget())
    assert layout.count() == 3
    assert layout.currentIndex() == 0


def test_scrolling(filledlayout):
    """ Tests for scrolling back and forth and wrap around """
    assert filledlayout.count() == 3
    filledlayout.scroll_back()
    filledlayout.scroll_back()
    assert filledlayout.currentIndex() == 1
    filledlayout.scroll_forth()
    assert filledlayout.currentIndex() == 2

    # Scroll over until after end of list
    filledlayout.scroll_forth()
    filledlayout.scroll_forth()
    assert filledlayout.currentIndex() == 1

    # Scroll back and roll over to top of list
    filledlayout.scroll_back()
    assert filledlayout.currentIndex() == 0

    # Scroll multiple
    filledlayout.scroll_somewhere(-2)
    assert filledlayout.currentIndex() == 1
    filledlayout.scroll_somewhere(2)
    assert filledlayout.currentIndex() == 0
