#!/usr/bin/env python3
from visualizer import utils
import tempfile


def test_imagegrabber_gets_any_extension():
    """
    @Linnea: W asks: does naming convention make any sense at all?
    Variable names for the test file structures are a bit like this:
    (Though the names when running are random; determined by 'tempfile')

    /tmp/
    ├── dir_100/
    │   ├── file_101
    │   ├── file_102
    │   ├── file_103
    │   └── dir_110/
    │       ├── file_111
    │       ├── file_112
    │       └── file_113
    └── dir_200/
        ├── file_201
        ├── file_202
        └── file_203
    """
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
    tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")
    tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")
    tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")

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

    # glob.glob returns list in arbitrary order, so compare without order
    assert set(grabbed) == set(goal)


def test_ignore_unimportant_files():
    dir_100 = tempfile.TemporaryDirectory()
    tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".pdf")
    tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".doc")
    tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".md")
    tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".spam")
    tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".eggs")
    tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".bacon")
    tempfile.NamedTemporaryFile(dir=dir_100.name, suffix=".spam")

    grabbed = utils.grab_image_paths_in_dir(dir_100.name)

    assert grabbed == []
