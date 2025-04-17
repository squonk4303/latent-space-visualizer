#!/usr/bin/env python3
"""Contains values meant to be accessible from anywhere."""
from collections import OrderedDict
from enum import Enum, auto
from pathlib import Path
import os

# Relevant Numbers
seed = int()  # useful value allocated in 'arguments.py'
STANDARD_IMG_SIZE = 640


# GUI Text
OPEN_FILE_LABEL = "&Open File"
APPLICATION_TITLE = "Latent Space Visualizer"
STAGE_TITLE = "Select Data"
GRAPH_TITLE = "Visualize Data"


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
    # fmt: off
    "#00008b", "#00008b", "#0000ff", "#0000ff", "#006400", "#006400", "#008b8b",
    "#008b8b", "#00fa9a", "#00fa9a", "#00ff00", "#00ff00", "#1e90ff", "#1e90ff",
    "#32cd32", "#4682b4", "#483d8b", "#696969", "#7b68ee", "#7f0000", "#7fffd4",
    "#800080", "#808000", "#8a2be2", "#8fbc8f", "#adff2f", "#afeeee", "#b03060",
    "#d8bfd8", "#da70d6", "#dc143c", "#f08080", "#f0e68c", "#f4a460", "#ff0000",
    "#ff0080", "#ff00ff", "#ff1493", "#ff8c00", "#ffd700",
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
    ("pickle", "Pickle Files (*.pickle *.pck *pcl)"),
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

# Disct for function selection 
# Add your desired function with the matched string here
functions = {
     "TSNE" : print,
     "PCA" : print,
     "UMAP" : print,
     "TRIMAP" : print,
     "PACMAP" : print,
     "SEGMENTATION" : print,
     "CLASSIFICATION" : print
}

# These are used in "getattr", so they're case sensitive, and it's
# important to keep them updated. Always run tests when refactoring models.
MODEL_TYPES = ["FCNResNet101"]


# Flags. If anyone asks why this is in consts, tell them to <class 'zip'> it
# Then consider whether to rename this file to "values.py" or something and
# make a class with consts instead or something , as well as store other
# values like flags, seeds, stubs, &c.
flags = {
    "dev": False,
    "truncate": False,
    "xkcd": False,
}
