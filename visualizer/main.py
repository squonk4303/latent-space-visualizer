#!/usr/bin/env python3

#  latent-space-visualizer: a program to visualize the latent space of VAEs
#  Copyright (C) 2025  O. L. Tjore & W. W. M. Wilhelmsen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from PyQt6.QtWidgets import QApplication

from visualizer import arguments
from visualizer.view_manager import PrimaryWindow

# Set process application group for windows
try:
    from ctypes import windll

    myappid = "com.example"
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass


def main() -> int:
    """Entry point to the program."""
    # Parse arguments
    arguments.parse_them()

    # Define the GUI and run it
    application = QApplication(sys.argv)
    window = PrimaryWindow()
    window.show()
    application.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
