#!/usr/bin/env python3
import numpy as np
import pytest
import torch
from visualizer import consts, loading, utils


@pytest.fixture
def model_obj():
    model_obj = loading.AutoencodeModel()
    return model_obj


@pytest.mark.pretrained_model
def test_load_model(model_obj):
    """Just runs this to see if it crashes."""
    # @Wilhelmsen: Is there anything more useful to test here?
    model = model_obj.load_model(consts.TRAINED_MODEL, ["skin"])
    assert model is not None


def test_dataset_to_tensor(model_obj):
    dataset = utils.grab_image_paths_in_dir(consts.SMALL_DATASET)
    tensors = model_obj.dataset_to_tensors(dataset)
    assert tensors is not None


def test_apply_tsne(model_obj):
    """Test features are reduced to desired dimensions and also it doesn't creash."""
    array = np.random.rand(8, 512)
    features = torch.Tensor(array)
    reduced = model_obj.apply_tsne(features, target_dimensions=2)
    assert reduced.shape[1] == 2
    reduced = model_obj.apply_tsne(features, target_dimensions=3)
    assert reduced.shape[1] == 3
