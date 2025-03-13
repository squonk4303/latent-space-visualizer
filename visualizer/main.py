#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import QApplication
from view_manager import PrimaryWindow
import arguments


def main() -> int:
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
