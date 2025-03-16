#!/usr/bin/env python3
from unittest.mock import patch
import pytest

from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
)

from visualizer import loading, utils
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


def test_some_path_set():
    """Test function sets path from return value."""
    with mocked_qfiledialog:
        path = open_dialog.for_some_file(parent=None)
        assert path == "a/gorgeous/path"


def test_img_path_set():
    """Test function sets path from return value."""
    with mocked_qfiledialog:
        path = open_dialog.for_image_file(parent=None)
        assert path == "a/gorgeous/path"


def test_model_path_set():
    """Test function sets path from return value."""
    with mocked_qfiledialog:
        path = open_dialog.for_trained_model_file(parent=None)
        assert path == "a/gorgeous/path"


# TEMP: Evaluating test_fixtures
import tempfile
import os


def test_1():
    """
    @Linnea: Question from W: does naming convention make sense?
    Variable names for the test file structures are a bit like this:
    (Though the names when running are random; determined by 'tempfile')

    /tmp
    ├── dir_100
    │   ├── file_101
    │   ├── file_102
    │   ├── file_103
    │   └── dir_110
    │       ├── file_111
    │       ├── file_112
    │       └── file_113
    └── dir_200
        ├── file_201
        ├── file_202
        └── file_203
    """

    dir_100 = tempfile.TemporaryDirectory()
    file_101 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".jpeg")
    file_102 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".jpeg")
    file_103 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".jpeg")

    dir_110 = tempfile.TemporaryDirectory(dir=dir_100.name)
    file_111 = tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")
    file_112 = tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")
    file_113 = tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")

    dir_200 = tempfile.TemporaryDirectory()
    file_201 = tempfile.NamedTemporaryFile(dir=dir_200.name)
    file_202 = tempfile.NamedTemporaryFile(dir=dir_200.name)
    file_203 = tempfile.NamedTemporaryFile(dir=dir_200.name)

    # Write something to file
    with open(file_101.name, "w") as f:
        f.write("Yahaha! You found me!")

    # Check contents of file
    with open(file_101.name) as f:
        content = f.read()
        # print(content)
        assert content == "Yahaha! You found me!"

    # Check ls of dir
    filepaths = [os.path.join(dir_100.name, f) for f in os.listdir(dir_100.name)]
    # print("FILES:", filepaths)


# Check that if the function runs in a populated directory, it gets all the image files
def test_imagegrabber_gets_any_extension():
    # TODO include other extensions
    dir_100 = tempfile.TemporaryDirectory()
    file_101 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".bmp")
    file_102 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".gif")
    file_103 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".jpeg")
    file_104 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".jpg")
    file_105 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".png")
    file_106 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".svg")
    file_107 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".tif")
    file_108 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".tiff")
    file_109 = tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".webp")

    dir_110 = tempfile.TemporaryDirectory(dir=dir_100.name)
    file_111 = tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")
    file_112 = tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")
    file_113 = tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")

    grabbed = utils.grab_image_paths_in_dir(dir_100.name)
    goal = [
        file_101.name,
        file_102.name,
        file_103.name,
        file_104.name,
        file_105.name,
        file_106.name,
        file_107.name,
        file_108.name,
        file_109.name,
    ]

    # glob returns list in arbitrary order, so compare without order
    assert set(grabbed) == set(goal)


def imagegrabber_gets_jpgs():
    pass


# Check that it can handle dirs outside of the repo
# ^^ What about dirs from protected areas? Raise exception on illegal dirs?
# Check that if the function runs in a populated directory, it ignores the non-image files
# Check that if the function runs in an unpopulated directory, it returns empty
# Check that the function gets from current directory and NOT subdirectories (recursively)
# Check that it doesn't follow symlinks (if it recurses)
# Check that it doesn't follow looping symlinks
# Check that it does with multiple filetypes at once
