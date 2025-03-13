#!/usr/bin/env python3
"""Make root of repo accessible for test modules."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
