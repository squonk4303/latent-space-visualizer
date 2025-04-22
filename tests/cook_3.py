#!/usr/bin/env python3
"""
Script which just starts cooking.

Intentionally omits "test" from filename to avoid being run automatically
by pytest. Run this script manually with pytest by specifying the file as
an argument... and run with "-s" to not suppress stdout. Does not display
the GUI on finish.

Cook (3 of 2) verb:
    running a version of our program with
    pre-determined parameters; for testing.

See also: "cook one's goose"
"""

import os

from visualizer import consts
from visualizer.view_manager import PrimaryWindow
from visualizer.models.segmentation import FCNResNet101


def test_cookin_brains(qtbot):
    window = PrimaryWindow()
    qtbot.addWidget(window)

    model = os.path.join(consts.REPO_DIR, "models.ignore/rgb-aug0/best_model.pth")
    window.data.model = FCNResNet101()
    window.data.model.load(model)
    window.data.layer = consts.LAYER
    window.data.dataset_location = os.path.join(consts.REPO_DIR, "pics/dataset_w_json")

    window.start_cooking_iii()
