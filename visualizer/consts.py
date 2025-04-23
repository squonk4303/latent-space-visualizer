#!/usr/bin/env python3
"""Contains values meant to be accessible from anywhere."""
from collections import OrderedDict
from enum import Enum, auto
from pathlib import Path
import os
from visualizer.models import segmentation

# Relevant Numbers
seed = int()  # useful value allocated in 'arguments.py'
STANDARD_IMG_SIZE = 640


# GUI Text

OPEN_FILE_LABEL = "&Open File"
APPLICATION_TITLE = "Latent Space Visualizer"
STAGE_TITLE = "Select Data"
GRAPH_TITLE = "Visualize Data"
DATASET_DIALOG_CAPTION = "SELECT DATASET"
IMAGE_DIALOG_CAPTION = "SELECT IMAGE"
LOAD_FILE_DIALOG_CAPTION = "SELECT FILE TO LOAD"
TRAINED_MODEL_DIALOG_CAPTION = "SELECT NEURAL-NET MODEL"
VERSION = "0.0.3"


# Plaintext with other uses
PROGRAM_DESCRIPTION = """
Launch a GUI which helps manipulate the latent space of a trained neural-network model.
And that kind of thing.
"""


# Colors
class COLOR(Enum):
    BACKGROUND = auto()
    TEXT = auto()


COLORS32 = [
    "#696969",
    "#006400",
    "#8b0000",
    "#808000",
    "#483d8b",
    "#008b8b",
    "#cd853f",
    "#000080",
    "#9acd32",
    "#7f007f",
    "#8fbc8f",
    "#b03060",
    "#ff0000",
    "#ff8c00",
    "#ffd700",
    "#7fff00",
    "#8a2be2",
    "#00ff7f",
    "#00ffff",
    "#00bfff",
    "#0000ff",
    "#ff6347",
    "#da70d6",
    "#b0c4de",
    "#ff00ff",
    "#1e90ff",
    "#f0e68c",
    "#90ee90",
    "#ff1493",
    "#7b68ee",
    "#fff8dc",
    "#ffb6c1",
]

COLORS16 = [
    # fmt: off
    "#2f4f4f", "#7f0000", "#006400", "#bdb76b", "#000080", "#ff0000", "#ffa500",
    "#ffff00", "#c71585", "#00ff00", "#00fa9a", "#00ffff", "#0000ff", "#d8bfd8",
    "#ff00ff","#1e90ff"
    # fmt: on
]


# File-paths &c.
# @Wilhelmsen: We should consider changing the valid image file extensions
# based on what is accepted by torch and matplotlib...
REPO_DIR = Path(os.path.dirname(__file__)).parent
SAVE_DIR = Path(REPO_DIR, "save_data")
QUICKSAVE_PATH = Path(SAVE_DIR, "quicksave.pickle")

# fmt: off
FILE_FILTERS = OrderedDict([
    ("whatever", "All Files (*)"),
    ("pickle", "Pickle Files (*.pickle *.pck *.pcl)"),
    ("pictures", "Image Files (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.svg)"),
    ("pytorch", "PyTorch Files (*.pt *.pth)"),
])
# fmt: off

# Used for testing only; assumes there's a local dir/symlink with appropriate model
GRAPHICAL_IMAGE = Path(REPO_DIR, "pics/crouton.png")
IMAGE_DATASET = Path(REPO_DIR, "pics/testing_dataset")
MEDIUM_DATASET = Path(REPO_DIR, "pics/medium_dataset")
MULTILABEL_MODEL = Path(REPO_DIR, "models.ignore/rgb-aug0/best_model.pth")
S_DATASET = Path(REPO_DIR, "models.ignore/dataset_w_json")
SMALL_DATASET = Path(REPO_DIR, "pics/small_dataset")
TRAINED_MODEL = Path(REPO_DIR, "models.ignore/RGB_no_augmentation.pth")

LAYER = "layer4"

# These are used in "getattr", so they're case sensitive, and it's
# important to keep them updated. Always run tests when refactoring models.
# Uses list comprehension to get all attributes of the module which are probably classes
# This is evil Python. Don't tell the cops
MODEL_TYPES = [c for c in dir(segmentation) if c[0].isupper()]


# Flags. If anyone asks why this is in consts, tell them to <class 'zip'> it
# Then consider whether to rename this file to "values.py" or something and
# make a class with consts instead or something , as well as store other
# values like flags, seeds, stubs, &c.
flags = {
    "dev": False,
    "truncate": False,
    "xkcd": False,
}
