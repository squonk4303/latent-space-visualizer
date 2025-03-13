#!/usr/bin/env python3
from unittest.mock import patch
import pytest

from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
)

import loading


# --- Fixtures and Sort-of-Fixtures ---
@pytest.fixture
def handler():
    window = QMainWindow
    return loading.FileDialogManager(window)


mocked_qfiledialog = patch.object(
    QFileDialog, "getOpenFileName", return_value=("a/gorgeous/path", "All Files (*)")
)


def test_default_file_dialogue(handler):
    """Test a normal use case for the function."""
    with mocked_qfiledialog:
        path, filter_ = handler.find_file()
        assert path == "a/gorgeous/path"
        assert filter_ == "All Files (*)"


def test_bad_file_filters(handler):
    """Test that a RuntimeError is raised upon bad list invocation."""
    with pytest.raises(RuntimeError):
        with mocked_qfiledialog:
            path, filter_ = handler.find_file(file_filters="I'm a chuckster!")


def test_some_path_set(handler):
    """Test function sets path from return value."""
    with mocked_qfiledialog:
        path = handler.find_some_file()
        assert path == "a/gorgeous/path"


def test_img_path_set(handler):
    """Test function sets path from return value."""
    with mocked_qfiledialog:
        path = handler.find_picture_file()
        assert path == "a/gorgeous/path"


def test_model_path_set(handler):
    """Test function sets path from return value."""
    with mocked_qfiledialog:
        path = handler.find_trained_model_file()
        assert path == "a/gorgeous/path"
