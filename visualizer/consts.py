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
TRAINED_MODEL = "models.ignore/RGB_no_augmentation.pth"
IMAGE_DATASET = "pics/testing_dataset"
SMALL_DATASET = "pics/small_dataset"
GRAPHICAL_IMAGE = "pics/1000.jpeg"
SEED = int()

# Flags. If anyone asks why this is in consts, tell them to <class 'zip'> it
# Then consider whether to rename this file to "values.py" or something and
# make a class with consts instead or something , as well as store other
# values like flags, seeds, stubs, &c.
flags = {
    "xkcd": False,
    "dev": False,
}
