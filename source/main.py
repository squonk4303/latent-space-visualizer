#!/usr/bin/env python3
import argparse
import sys
from PyQt6.QtWidgets import QApplication
from view_manager import PrimaryWindow
import consts
import utils

if __name__ == "__main__":
    # Set up for command-line arguments
    parser = argparse.ArgumentParser(description=consts.PROGRAM_DESCRIPTION)

    # Specify command-line arguments
    parser.add_argument(
        "--xkcd", help="display the plot in a different style", action="store_true"
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="set seed for all relevant packages"
    )

    args = parser.parse_args()

    # Handle arguments
    consts.flags["xkcd"] = args.xkcd

    if args.seed is not None:
        utils.superseed(int(args.seed))

    # Define the GUI and run it
    app = QApplication(sys.argv)
    window = PrimaryWindow()
    window.show()
    app.exec()
