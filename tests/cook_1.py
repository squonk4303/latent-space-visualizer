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

from torchvision.models import resnet101, ResNet101_Weights
from visualizer.view_manager import PrimaryWindow


def test_cookin_brains(qtbot):
    window = PrimaryWindow()
    qtbot.addWidget(window)

    window.data.model = resnet101(weights=ResNet101_Weights.DEFAULT)
    window.data.model.eval()
    window.data.layer = "layer4"

    window.start_cooking_brains()

    # Assert that some filepaths are found and placed in a dataset structure
    # ^^ same with labels (unsure about with image data)
    # May want to transfer as a test for the dataset structure itself


# @Wilhelmsen: Yet to test that quickloading plots its data
