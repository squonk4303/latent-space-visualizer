#!/usr/bin/env python3
"""Contains values meant to be accessible from anywhere."""
import os
from enum import Enum, auto

# Relevant Numbers
seed = int()  # useful value allocated in 'arguments.py'
STANDARD_IMG_SIZE = 640


# GUI Text
OPEN_FILE_LABEL = "&Open File"
WINDOW_TITLE = "Latent Space Visualizer"


# Plaintext with other uses
PROGRAM_DESCRIPTION = """
Launch a GUI which helps manipulate the latent space of a trained neural-network model.
And that kind of thing.
"""


# File-paths &c.
# @Wilhelmsen: We should consider changing the valid image file extensions
# based on what is accepted by torch and matplotlib...
BASE_MODULE_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(BASE_MODULE_DIR, os.pardir))

SAVE_DIR = os.path.join(REPO_DIR, "save_data")
QUICKSAVE_PATH = os.path.join(SAVE_DIR, "quicksave.pickle")

FILE_FILTERS = {
    "pictures": "Image Files (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.svg)",
    "pytorch": "PyTorch Files (*.pt *.pth)",
    "pickle": "Pickle Files (*.pickle *.pck *pcl)",
    "whatever": "All Files (*)",
}

# Used for testing only; assumes there's a local dir/symlink with appropriate model
TRAINED_MODEL = os.path.join(REPO_DIR, "models.ignore/RGB_no_augmentation.pth")
IMAGE_DATASET = os.path.join(REPO_DIR, "pics/testing_dataset")
SMALL_DATASET = os.path.join(REPO_DIR, "pics/small_dataset")
MEDIUM_DATASET = os.path.join(REPO_DIR, "pics/medium_dataset")
GRAPHICAL_IMAGE = os.path.join(REPO_DIR, "pics/crouton.png")

SINE_CSV = os.path.join(REPO_DIR, "source/sine.csv")
SINE_COSINE = os.path.join(REPO_DIR, "source/sin_cos.csv")

DEFAULT_MODEL_CATEGORIES = ["skin"]
DEVICE = None

# Enums for dimensionality reduction techniques
class DR_technique(Enum):
    T_SNE = auto() 
    PCA = auto()
    UMAP = auto()
    TRIMAP = auto()
    PACMAP = auto()

# Flags. If anyone asks why this is in consts, tell them to <class 'zip'> it
# Then consider whether to rename this file to "values.py" or something and
# make a class with consts instead or something , as well as store other
# values like flags, seeds, stubs, &c.
flags = {
    "xkcd": False,
    "dev": False,
}
