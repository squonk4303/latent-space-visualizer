#!/usr/bin/env python3
import dataclasses
import pytest
import torch
from visualizer import consts
from visualizer.plottables import Plottables


# --- Fixtures ---
@pytest.fixture
def data_object():
    data = Plottables()
    # data.model = FCNResNet101(["skin"])
    # data.model.load(consts.TRAINED_MODEL)
    data.dataset_plottable = torch.tensor(
        [
            [0.18295779, 0.42863305],
            [0.71485087, 0.04020805],
            [0.88153443, 0.34253962],
            [0.79842691, 0.02809093],
        ]
    )
    return data


def test_assert_dataclass_equivalence(data_object):
    # Define data1 and data2 as Plottables with equivalent values
    data1 = Plottables(**dataclasses.asdict(data_object))
    data2 = Plottables(**dataclasses.asdict(data_object))

    assert data1 == data2
    data2.dataset_plottable = None
    assert data1 != data2


def test_dataclass_identity(data_object):
    # Define data1 and data2 as Plottables with equivalent values
    data1 = Plottables(**dataclasses.asdict(data_object))
    data2 = Plottables(**dataclasses.asdict(data_object))

    assert data1 is not data2
    data1 = data2
    assert data1 is data2
    data1.dataset_plottable = None
    assert data1 == data2
