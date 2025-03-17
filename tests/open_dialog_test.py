#!/usr/bin/env python3
from unittest.mock import patch
import pytest
from PyQt6.QtWidgets import QFileDialog
from visualizer import open_dialog


# --- Fixtures and Sort-of-Fixtures ---
mocked_qfiledialog = patch.object(
    QFileDialog, "getOpenFileName", return_value=("a/gorgeous/path", "All Files (*)")
)


def test_default_file_dialogue():
    """Test a normal use case for the function."""
    with mocked_qfiledialog:
        path, filter_ = open_dialog.for_file(parent=None)
        assert path == "a/gorgeous/path"
        assert filter_ == "All Files (*)"


def test_bad_file_filters():
    """Test that a RuntimeError is raised upon bad list invocation."""
    with pytest.raises(RuntimeError):
        with mocked_qfiledialog:
            path, filter_ = open_dialog.for_file(
                parent=None, file_filters="I'm a chuckster!"
            )


def test_some_file_dialog():
    """These tests are really just here in case something breaks within the functions."""
    with mocked_qfiledialog:
        path = open_dialog.for_some_file(parent=None)
        assert path == "a/gorgeous/path"


def test_image_file_dialog():
    with mocked_qfiledialog:
        path = open_dialog.for_image_file(parent=None)
        assert path == "a/gorgeous/path"


def test_model_file_dialog():
    with mocked_qfiledialog:
        path = open_dialog.for_trained_model_file(parent=None)
        assert path == "a/gorgeous/path"
