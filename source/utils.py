#!/usr/bin/env python3
import numpy as np
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
