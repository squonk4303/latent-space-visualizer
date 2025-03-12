#!/usr/bin/env python3
import argparse
import sys
from PyQt6.QtWidgets import QApplication
from view_manager import PrimaryWindow
import consts

if __name__ == "__main__":
    # Set up for command-line arguments
    parser = argparse.ArgumentParser(description=consts.PROGRAM_DESCRIPTION)

    # Specify command-line arguments
    parser.add_argument(
        "--xkcd", help="display the plot in a different style", action="store_true"
    )

    args = parser.parse_args()
    consts.flags["xkcd"] = args.xkcd

    # Define the GUI and run it
    app = QApplication(sys.argv)
    window = PrimaryWindow()
    window.show()
    app.exec()
