#!/usr/bin/env python3

class ConstClass():
    """
    A class for holding constant values consistenly throughout the program
    """
    def __init__(self):
        self.FILE_FILTERS = [
            "All Files (*.*)",
            "PyTorch Files (*.pt)",
        ]
        self.OPEN_FILE_LABEL = "&Open File"
        self.STATUS_TIP_TEMP = "TODO: What's a good status tip guide?"
        self.WINDOW_TITLE = "Latent Space Visualizer"


Const = ConstClass()
