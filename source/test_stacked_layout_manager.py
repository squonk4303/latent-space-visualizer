#!/usr/bin/env python3
import pytest

from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLayout,
    QVBoxLayout,
    QWidget,
)

from stacked_layout_manager import StackedLayoutManager


# Qt requires appeasal by constructing a QApplication before any QWidgets
# Lest it sends a REALLY nasty-looking bug at you. Not recommended.
_ = QApplication([])


@pytest.fixture
def empty_layout():
    """Initialize a layout manager with no widgets."""
    layout = StackedLayoutManager()
    return layout


@pytest.fixture
def filled_layout():
    """Initialize a layout manager with 3 arbitrary widgets."""
    widgets = (QWidget(), QWidget(), QWidget())
    layout = StackedLayoutManager(widgets)
    return layout


def test_layout_manager_initiation(empty_layout):
    """ Tests initialization of a layout """
    assert empty_layout.count() == 0
    assert empty_layout.currentWidget() is None
    assert empty_layout.currentIndex() == -1


def test_adding_layers(empty_layout):
    """
    Tests adding QHBoxLayouts by method. QHBoxlayouts are just examples of
    layouts. This should be valid for all layouts except QLayouts.
    See below error test.
    """
    layouts = (QHBoxLayout(), QVBoxLayout(), QHBoxLayout())
    empty_layout.add_layout(layouts[0])
    empty_layout.add_layout(layouts[1])
    empty_layout.add_layout(layouts[2])
    assert empty_layout.count() == 3
    assert empty_layout.currentIndex() == 0


def test_alt_initalization_with_qlayouts_error(empty_layout):
    """
    Tries to initiate QLayouts by list argument, and fails at putting QLayouts in a list.
    NOTE: This fails because QLayouts are a C++ abstract class
    """
    with pytest.raises(TypeError):
        layouts = (QLayout(), QLayout(), QLayout())
        layout = StackedLayoutManager(layouts)
        assert layout.count() == 3
        assert layout.currentlayout() == layouts[0]
        assert layout.currentIndex() == 0


def test_tuple_initialization_then_method_adding(filled_layout):
    """
    Tests for whether initializing with a tuple destroys
    the capability to append more widgets by method
    """
    assert filled_layout.count() == 3
    filled_layout.addWidget(QWidget())
    assert filled_layout.count() == 4


def test_adding_widgets():
    """ Tests adding widgets to the layoutmanager individually by method """
    layout = StackedLayoutManager()
    layout.add_widget(QWidget())
    layout.add_widget(QWidget())
    layout.add_widget(QWidget())
    assert layout.count() == 3
    assert layout.currentIndex() == 0


def test_scrolling(filled_layout):
    """ Tests for scrolling back and forth and wrap around """
    assert filled_layout.count() == 3
    filled_layout.scroll_back()
    filled_layout.scroll_back()
    assert filled_layout.currentIndex() == 1
    filled_layout.scroll_forth()
    assert filled_layout.currentIndex() == 2

    # Scroll over until after end of list
    filled_layout.scroll_forth()
    filled_layout.scroll_forth()
    assert filled_layout.currentIndex() == 1

    # Scroll back and roll over to top of list
    filled_layout.scroll_back()
    assert filled_layout.currentIndex() == 0

    # Scroll multiple
    filled_layout.scroll_somewhere(-2)
    assert filled_layout.currentIndex() == 1
    filled_layout.scroll_somewhere(2)
    assert filled_layout.currentIndex() == 0
