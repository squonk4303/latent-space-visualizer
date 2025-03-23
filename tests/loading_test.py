#!/usr/bin/env python3
import numpy as np
import pytest
import torch
from visualizer import consts, loading, utils
from visualizer.external.fcn import FCNResNet101


@pytest.mark.pretrained_model
def test_load_model():
    """Just runs this to see if it crashes."""
    # @Wilhelmsen: Is there anything more useful to test here?
    loading.ensure_device()
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
