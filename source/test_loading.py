#!/usr/bin/env python3
from unittest.mock import patch
import pytest

from PyQt6.QtWidgets import QFileDialog

import consts
import loading


# --- Sort-of-Fixtures ---
mocked_qfiledialog = patch.object(
        QFileDialog,
        "getOpenFileName",
        return_value=("a/gorgeous/path", "All Files (*)")
    )


def test_default_file_dialogue():
    """Test a normal use case for the function."""
    handler = loading.FileDialogManager()
    with mocked_qfiledialog:
        path, filter_ = handler.open_dialogue()
        assert path    == "a/gorgeous/path"
        assert filter_ == "All Files (*)"


def test_bad_file_filters():
    """Test that a RuntimeError is raised upon bad list invocation."""
    handler = loading.FileDialogManager()
    with pytest.raises(RuntimeError):
        with mocked_qfiledialog:
            path, filter_ = handler.open_dialogue(file_arr="I'm a chuckster!")


def test_some_path_set():
    """Test function sets path from return value."""
    handler = loading.FileDialogManager()
    with mocked_qfiledialog:
        handler.open_all()
        assert handler.some_path == "a/gorgeous/path"


def test_img_path_set():
    """Test function sets path from return value."""
    handler = loading.FileDialogManager()
    with mocked_qfiledialog:
        handler.open_img()
        assert handler.img_path == "a/gorgeous/path"


def test_model_path_set():
    """Test function sets path from return value."""
    handler = loading.FileDialogManager()
    with mocked_qfiledialog:
        handler.open_model()
        assert handler.model_path == "a/gorgeous/path"
