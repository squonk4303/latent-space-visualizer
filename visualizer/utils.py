#!/usr/bin/env python3
import glob
import numpy as np
import os
import torch


def arr_is_subset(arr1, arr2):
    """Compare if arr1 is part of arr2, return bool."""
    return set(arr1) <= set(arr2)


def superseed(seed):
    """
    Set seed for numpy and torch.

    This WILL have to expand if we introduce packages with other random
    number generators. We could also have this function set a variable in
    consts, so that the seed is available for grabbing, like having it
    displayed on-screen so the user and others can know what seed a run is
    based on. Hey that's a good idea. It's just like the binding of isaac.
    Note though that forcing determinism may decrease performance.

    For assured determinism, consider also applying
    `torch.use_deterministic_algorithms()`.
    See more at ()[https://pytorch.org/docs/stable/notes/randomness.html]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def grab_image_paths_in_dir(dir_path):
    """
    Return all image files found in a directory.

    Specifically returns a list containing absolute filepaths.
    Does not search through subdirectories or symlinks.

    @Wilhelmsen: Consider iglob; it makes an iterator, which would save memory with large datasets
    @Wilhelmsen: consider adding recursive option, this also need implementation in the interface
    """
    # Make strings such as "/over/hills/far/away/*.jpeg", using dir_path
    patterns = [
        os.path.join(dir_path, "*.bmp"),
        os.path.join(dir_path, "*.gif"),
        os.path.join(dir_path, "*.jpeg"),
        os.path.join(dir_path, "*.jpg"),
        os.path.join(dir_path, "*.png"),
        os.path.join(dir_path, "*.svg"),
        os.path.join(dir_path, "*.tif"),
        os.path.join(dir_path, "*.tiff"),
        os.path.join(dir_path, "*.webp"),
    ]

    # The list comprehension statement here makes a nested list,
    # and 'sum' is used here to flatten that list
    filepaths = sum([glob.glob(pattern, root_dir=dir_path) for pattern in patterns], [])
    return filepaths
