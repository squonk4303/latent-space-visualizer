#!/usr/bin/env python3

# GUI Text
OPEN_FILE_LABEL = "&Open File"
WINDOW_TITLE = "Latent Space Visualizer"

# Plaintext with other uses
PROGRAM_DESCRIPTION = """
Launch a GUI which helps manipulate the latent space of a trained neural-network model.
And that kind of thing.
"""

# File-paths &c.
FILE_FILTERS = {
    "pictures": "Image Files (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.svg)",
    "pytorch": "PyTorch Files (*.pt *.pth)",
    "whatever": "All Files (*)",
}

SINE_CSV = "source/sine.csv"
SINE_COSINE = "source/sin_cos.csv"

# Used for testing only; assumes there's a local dir/symlink with appropriate model
TRAINED_MODEL = "trained_models/RGB_no_augmentation.pth"
GRAPHICAL_IMAGE = "pics/dog.png"
SEED = 42

# Flags. If anyone asks why this is in consts, tell them to <class 'zip'> it
flags = {
    "xkcd": False,
}
