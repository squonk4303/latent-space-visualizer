#!/usr/bin/env python3
import pytest
from PyQt6.QtCore import Qt
from window import SomeWindow


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
