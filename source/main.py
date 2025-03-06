#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import QApplication
from view_manager import PrimaryWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PrimaryWindow()
    window.show()
    app.exec()
