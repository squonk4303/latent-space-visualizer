#!/usr/bin/env python3
import numpy as np
import pytest
import torch

from visualizer import loading, utils


@pytest.fixture()
def set_seed():
    """Apply this seed to all subsequent tests."""
    utils.superseed(327)


def test_setup_np_array(set_seed):
    """Just set up array. Don't assert anything."""
    global examinee_array
    examinee_array = np.random.rand(256, 256)


def test_array_same_w_seed(set_seed):
    """Assert whether all elements of two arrays built on the same seed are equal."""
    examiner_array = np.random.rand(256, 256)
    assert np.array_equal(examinee_array, examiner_array)


def test_array_different_w_o_seed():
    """
    Assert whether all elements of two arrays not built on the same seed are equal.

    As you might expect, there's a small chance this function yields a false negative.
    """
    examiner_array = np.random.rand(256, 256)
    assert not np.array_equal(examinee_array, examiner_array)


def test_setup_tsne(set_seed):
    global examinee_array
    tensor = torch.Tensor(np.random.rand(3, 512))
    examinee_array = loading.apply_tsne(tensor)


def test_tsne_same_with_seed(set_seed):
    tensor = torch.Tensor(np.random.rand(3, 512))
    examiner_array = loading.apply_tsne(tensor)
    assert np.array_equal(examiner_array, examinee_array)


def test_tsne_different_w_o_seed():
    tensor = torch.Tensor(np.random.rand(3, 512))
    examiner_array = loading.apply_tsne(tensor)
    assert not np.array_equal(examiner_array, examinee_array)


# reproducibility TODOs:
# - Include determinism check with torch.use_deterministic_algorithms
# - Include checks for this which misfire
# - Include tests for torch-based RNG too
