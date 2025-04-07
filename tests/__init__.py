#!/usr/bin/env python3
"""Make root of repo accessible for test modules."""
import os
import sys
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Qt requires appeasal by constructing a QApplication before any QWidgets
# Lest it sends a REALLY nasty-looking bug at you. Not recommended.
_ = QApplication([])
