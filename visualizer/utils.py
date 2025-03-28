#!/usr/bin/env python3
import glob
import numpy as np
import os
import torch

from visualizer import consts


def arr_is_subset(arr1, arr2):
    """Compare if arr1 is part of arr2, return bool."""
    return set(arr1) <= set(arr2)


def superseed(seed):
    """
    Set seed for numpy and torch.

    Used in module 'arguments'
    Sets 'consts.seed' because it's used by our t-SNE functions.

    This WILL have to expand if we introduce packages with other random
    number generators. We could also have this function set a variable in
    consts, so that the seed is available for grabbing, like having it
    displayed on-screen so the user and others can know what seed a run is
    based on. Hey that's a good idea. It's just like the binding of isaac.
    Note though that forcing determinism may decrease performance.
    Note also that determinism matches well with caching.

    For assured determinism, consider also applying
    `torch.use_deterministic_algorithms()`.
    See more at ()[https://pytorch.org/docs/stable/notes/randomness.html]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    consts.seed = seed


def grab_image_paths_in_dir(dir_path, *, recursive=False):
    """
    Return all image files found in a directory.

    This is how the program works; can be changed:
    - Specifically returns a list containing absolute filepaths
    - Does not search through subdirectories or symlinks
    - Ignores hidden files

    @Wilhelmsen: Consider a variation using iglob; it makes an iterator,
                 which would save memory with large datasets Though,
                 it *is* just a list of strings...
    @Wilhelmsen: Let dir_path be a list *args maybe...
    """
    extensions = [
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".svg",
        ".tif",
        ".tiff",
        ".webp",
    ]

    # Make strings such as "/over/hills/far/away/*.jpeg", for dir_path "/over/hills/far/away/"
    # For some reason *does* require both the **/* and recursive=True in glob.glob
    if recursive:
        patterns = [os.path.join(dir_path, f"**/*{ex}") for ex in extensions]
    else:
        patterns = [os.path.join(dir_path, f"*{ex}") for ex in extensions]

    # The list comprehension statement here makes a nested list,
    # and 'sum(list, [])' is used here to flatten that list
    filepaths = sum(
        [
            glob.glob(pattern, root_dir=dir_path, recursive=recursive)
            for pattern in patterns
        ],
        [],
    )

    return filepaths
