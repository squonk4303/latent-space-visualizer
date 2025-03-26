#!/usr/bin/env python3
import numpy as np
import os
import pytest
import tempfile
import torch

from visualizer import consts, loading, utils
from visualizer.external.fcn import FCNResNet101
from visualizer.plottables import Plottables


# --- Fixtures ---
@pytest.fixture
def data_object():
    consts.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Plottables()
    data.model = FCNResNet101(["skin"])
    data.model.load(consts.TRAINED_MODEL)
    data.image_plottable = np.array(
        [
            [0.18295779, 0.42863305],
            [0.71485087, 0.04020805],
            [0.88153443, 0.34253962],
            [0.79842691, 0.02809093],
        ]
    )
    return data


@pytest.mark.require_pretrained_model
def test_load_model():
    """Just runs this to see if it crashes."""
    # @Wilhelmsen: Is there anything more useful to test here?
    consts.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNResNet101(["skin"])
    model.load(consts.TRAINED_MODEL)
    assert model is not None


def test_dataset_to_tensor():
    dataset = utils.grab_image_paths_in_dir(consts.SMALL_DATASET)
    tensors = loading.dataset_to_tensors(dataset)
    assert tensors is not None


def test_apply_tsne():
    """Test features are reduced to desired dimensions and also it doesn't creash."""
    array = np.random.rand(8, 512)
    features = torch.Tensor(array)
    reduced = loading.apply_tsne(features, target_dimensions=2)
    assert reduced.shape[1] == 2
    reduced = loading.apply_tsne(features, target_dimensions=3)
    assert reduced.shape[1] == 3


@pytest.mark.slow
def test_saving_and_loading_in_place(data_object):
    """
    Makes an object with some data, then saves the data,
    uses another object to load it, and compares the former with the latter.
    """
    temp_file = tempfile.NamedTemporaryFile(dir=consts.SAVE_DIR)
    loading.quicksave(data_object, temp_file.name)

    other_data = loading.quickload(temp_file.name)
    assert np.array_equal(data_object.image_plottable, other_data.image_plottable)


@pytest.mark.slow
def _test_save_to_persistent_file(data_object):
    persistent_file = os.path.join(consts.SAVE_DIR, "test_save.pickle")
    loading.quicksave(data_object, persistent_file)


@pytest.mark.slow
def test_loading_cold_model_file(data_object):
    persistent_file = os.path.join(consts.SAVE_DIR, "test_save.pickle")
    other_data = loading.quickload(persistent_file)
    assert np.array_equal(other_data.image_plottable, data_object.image_plottable)
