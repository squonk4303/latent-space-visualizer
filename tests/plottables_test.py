#!/usr/bin/env python3
import dataclasses
import pytest

from torchvision.models import resnet101, ResNet101_Weights

from visualizer import consts
from visualizer.plottables import SavableData


# --- Fixtures ---


@pytest.fixture
def data_object():
    data = SavableData()
    weights = ResNet101_Weights.DEFAULT
    data.model = resnet101(weights=weights)
    data.model.eval()
    data.layer = consts.LAYER
    data.dataset_location = consts.MEDIUM_DATASET
    # data.paths
    return data


# --- Tests ---


def test_assert_dataclass_equivalence(data_object):
    # Define data1 and data2 as SavableData with equivalent values
    data1 = SavableData(**dataclasses.asdict(data_object))
    data2 = SavableData(**dataclasses.asdict(data_object))
    assert data1 == data2
    # Assert they're not references to the same object
    assert data1 is not data2
    data2.model = None
    assert data1 != data2


def test_dataclass_identity(data_object):
    # Define data1 and data2 as SavableData with equivalent values
    data1 = SavableData(**dataclasses.asdict(data_object))
    data2 = SavableData(**dataclasses.asdict(data_object))

    assert data1 is not data2
    data1 = data2
    assert data1 is data2
    data1.dataset_plottable = None
    assert data1 == data2
