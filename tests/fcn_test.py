#!/usr/bin/env python3
import os
import pickle
import pytest
import torch
from visualizer import consts
from visualizer.external.fcn import FCNResNet101

FILE = os.path.join(consts.SAVE_DIR, "fcn.pth")


def _write_object_to_file():
    """Note that this isn't run when pytest is called."""
    consts.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNResNet101(consts.DEFAULT_MODEL_CATEGORIES)
    model.load(consts.TRAINED_MODEL)

    parent_dir = os.path.abspath(os.path.join(FILE, os.pardir))
    os.makedirs(parent_dir, exist_ok=True)

    with open(FILE, "wb") as f:
        pickle.dump(model, f)


@pytest.mark.require_pretrained_model
@pytest.mark.slow
@pytest.mark.stub
def test_loaded_model_equal_to_stub():
    """
    Test for whether the FCNResNet101 model is really the same as before.

    Does this by comparing it to an earlier iteration of which.
    Useful when refactoring the class without intending to actually change its effects.
    """
    consts.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fresh_model = FCNResNet101(consts.DEFAULT_MODEL_CATEGORIES)
    fresh_model.load(consts.TRAINED_MODEL)

    with open(FILE, "rb") as f:
        stubbed_model = pickle.load(f)

    # Naive assertion
    assert repr(fresh_model) == repr(stubbed_model)

    # Asserting all values in all tensors are equal
    for a, b in zip(fresh_model.parameters(), stubbed_model.parameters()):
        # print(f"{a.shape}\n{b.shape}\n")  # Print format for debuggin
        assert torch.equal(a, b)
