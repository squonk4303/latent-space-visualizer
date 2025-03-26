#!/usr/bin/env python3
import pickle
import torch
import pytest
from visualizer import consts
from visualizer.external.fcn import FCNResNet101

FILE = "stubs.ignore/fcn.pth"


def write_object_to_file():
    """Note that this isn't run when pytest is called."""
    consts.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNResNet101(["skin"])
    model.load(consts.TRAINED_MODEL)
    with open(FILE, "wb") as f:
        pickle.dump(model, f)


@pytest.mark.require_pretrained_model
@pytest.mark.require_stub
def test_loaded_model_equal_to_stub():
    consts.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fresh_model = FCNResNet101(["skin"])
    fresh_model.load(consts.TRAINED_MODEL)

    with open(FILE, "rb") as f:
        stubbed_model = pickle.load(f)

    # Naive assertion
    assert repr(fresh_model) == repr(stubbed_model)

    # Asserting all values in all tensors are equal
    for a, b in zip(fresh_model.parameters(), stubbed_model.parameters()):
        # print(f"{a.shape}\n{b.shape}\n")  # Print format for debuggin
        assert torch.equal(a, b)
