#!/usr/bin/env python3
from visualizer import utils
import tempfile


class TestImageGrabber:
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
    file_111 = tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")
    file_112 = tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")
    file_113 = tempfile.NamedTemporaryFile(dir=dir_110.name, suffix=".jpeg")

    dir_200 = tempfile.TemporaryDirectory()
    file_201 = tempfile.NamedTemporaryFile(dir=dir_200.name, suffix=".bmp")
    file_202 = tempfile.NamedTemporaryFile(dir=dir_200.name, suffix=".gif")
    file_203 = tempfile.NamedTemporaryFile(dir=dir_200.name, suffix=".jpeg")

    dir_900 = tempfile.TemporaryDirectory()
    file_901 = tempfile.NamedTemporaryFile(dir=dir_900.name)
    file_902 = tempfile.NamedTemporaryFile(dir=dir_900.name, suffix=".txt")
    file_903 = tempfile.NamedTemporaryFile(dir=dir_900.name, suffix=".pdf")
    file_904 = tempfile.NamedTemporaryFile(dir=dir_900.name, suffix=".doc")
    file_905 = tempfile.NamedTemporaryFile(dir=dir_900.name, suffix=".md")
    file_906 = tempfile.NamedTemporaryFile(dir=dir_900.name, suffix=".spam")
    file_907 = tempfile.NamedTemporaryFile(dir=dir_900.name, suffix=".eggs")
    file_908 = tempfile.NamedTemporaryFile(dir=dir_900.name, suffix=".bacon")
    file_909 = tempfile.NamedTemporaryFile(dir=dir_900.name, suffix=".spam")

    def test_imagegrabber_gets_any_extension(self):
        grabbed = utils.grab_image_paths_in_dir(self.dir_100.name)
        goal = [
            # fmt: off
            self.file_101.name, self.file_102.name, self.file_103.name,
            self.file_104.name, self.file_105.name, self.file_106.name,
            self.file_107.name, self.file_108.name, self.file_109.name,
            # fmt: on
        ]

        # glob.glob returns list in arbitrary order, so compare without order
        assert set(grabbed) == set(goal)

    def test_ignore_unimportant_files(self):
        grabbed = utils.grab_image_paths_in_dir(self.dir_900.name)
        # Using 'set()' as opposed to '{}' seems inconsistent with python
        # convention, BUT Using '{}' makes an empty dict, so the docs
        # recommend using the empty set constructor.
        # src: https://docs.python.org/3/tutorial/datastructures.html#sets
        assert set(grabbed) == set()

    def test_recursive(self):

        grabbed = utils.grab_image_paths_in_dir(self.dir_100.name, recursive=True)
        goal = [
            # fmt: off
            self.file_101.name, self.file_102.name, self.file_103.name,
            self.file_104.name, self.file_105.name, self.file_106.name,
            self.file_107.name, self.file_108.name, self.file_109.name,

            self.file_111.name, self.file_112.name, self.file_113.name,
            # fmt: on
        ]

        assert set(grabbed) == set(goal)
