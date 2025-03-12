#!/usr/bin/env python3
import random

import numpy as np
import torch


def arr_is_subset(arr1, arr2):
    """Compare if arr1 is part of arr2, return bool."""
    return set(arr1) <= set(arr2)


def superseed(seed):
    """Set seed for python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
