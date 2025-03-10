#!/usr/bin/env python3
from enum import Enum, auto

# GUI Text
OPEN_FILE_LABEL = "&Open File"
STATUS_TIP_TEMP = "TODO: What's a good status tip guide?"
WINDOW_TITLE    = "Latent Space Visualizer"

# File Paths et.c.
FILE_FILTERS = {
    "pictures": ("Image Files "
                 "(*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.svg)"),
    "pytorch":  "PyTorch Files (*.pt *.pth)",
    "whatever": "All Files (*)",
}


SINE_CSV      = "source/sine.csv"
SINE_COSINE   = "source/sin_cos.csv"

# Used for testing only; assumes there's a local dir/symlink with appropriate model
TRAINED_MODEL = "trained_models/RGB_no_augmentation.pth"

# Enums for dimensionality reduction techniques
class DR_technique(Enum):
    T_SNE = auto() 
    PCA = auto()
    UMAP = auto()
    TRIMAP = auto()
    PACMAP = auto()
