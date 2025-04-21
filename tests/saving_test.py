#!/usr/bin/env python3
from pathlib import Path
from unittest.mock import patch
import pickle

from PyQt6.QtWidgets import QFileDialog
import numpy as np
import PIL
import pytest
import torch

from visualizer import consts, utils
from visualizer.models.segmentation import FCNResNet101
from visualizer.plottables import SavableData
from visualizer.view_manager import dim_reduction_techs, PrimaryWindow


# =================
# Consts & Fixtures
# =================

SAVE_LOCATION = Path(consts.REPO_DIR, "stubs/save_integrity_file.pickle")
RESOLUTION = 28
SIZE = 10


def get_savable():
    data = SavableData()
    data.model = FCNResNet101()
    data.model_location = consts.MULTILABEL_MODEL
    data.model.load(data.model_location)
    data.layer = consts.LAYER
    data.dim_reduction = list(dim_reduction_techs)[0]
    data.dataset_location = consts.MEDIUM_DATASET

    # May want to use realer examples
    data.dataset_intermediary = np.random.rand(RESOLUTION, RESOLUTION)
    # Assure there's >SIZE valid categories, but truncate it to SIZE also
    data.labels = (data.model.categories * SIZE)[:SIZE]
    data.masks = PIL.Image.fromarray(data.dataset_intermediary)
    data.paths = utils.grab_image_paths_in_dir(data.dataset_location)[:SIZE]
    data.two_dee = torch.rand(SIZE, 2)
    return data


@pytest.fixture
def window(qtbot):
    w = PrimaryWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def obj_fixture():
    return get_savable()


# =========
# Functions
# =========


def save_savable():
    """Just saves to file, yo!"""
    data_obj = get_savable()
    directory = SAVE_LOCATION.parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(SAVE_LOCATION, "wb") as f:
        pickle.dump(data_obj, f)
        print(f"Saved file to {SAVE_LOCATION}, yo!")


def load_savable() -> SavableData:
    with open(SAVE_LOCATION, "rb") as f:
        data_obj = pickle.load(f)

    return data_obj


def assert_integrity_of(canary: SavableData):
    """
    Look for changes in SavableData.

    May notice:
    - Addition of attributes
    - Removal of attributes
    - Renaming of attributes

    May not notice:
    - Change in attribute type

    If this causes a test to fail, it may be because of a change in
    SavableData, meaning the next update will be a breaking change. Also
    obj_fixture should be updated and that stubs should be regenerated.
    """
    # Infers attributes from whether its first character is a letter
    # Note that dir() is not *entirely* reliable for finding every attribute
    # see: https://docs.python.org/3/library/functions.html#dir
    attr_names = [a for a in dir(canary) if a[0].isalpha()]
    # Assert that the attributes are the same as the ones defined in SavableData
    assert set(attr_names) == {a for a in dir(SavableData()) if a[0].isalpha()}
    for attribute in (getattr(canary, a) for a in attr_names):
        # NOTE: Is not able to assert empty strings
        assert attribute is not None
        # If it's a list, assert that *that's* not empty
        # note: Python is a little fussy about lists
        if isinstance(attribute, list):
            assert len(attribute) > 0


# =====
# Tests
# =====


@pytest.mark.stub
def test_the_object(obj_fixture):
    """
    Test the object filled here. If this test trips, saves may be broken.

    Also tests test-code, which may be a little paranoid,
    but I'm leaving this until there's a compelling reason to remove it.
    And it does help when trying something different with the assertions.
    """
    assert_integrity_of(obj_fixture)
    bad_object = obj_fixture
    bad_object.labels = []
    with pytest.raises(AssertionError):
        assert_integrity_of(bad_object)
    bad_object = obj_fixture
    bad_object.layer = None
    with pytest.raises(AssertionError):
        assert_integrity_of(bad_object)
    bad_object = obj_fixture
    bad_object.bogus = "Something"
    with pytest.raises(AssertionError):
        assert_integrity_of(bad_object)


@pytest.mark.stub
def test_compare_with_old_save(obj_fixture):
    """
    Compares current savable-object with the one that's stubbed.
    """
    old_obj = load_savable()
    assert obj_fixture == old_obj


# @Wilhelmsen: If we want to test for accesses to each attribute, we can do
# it here, granted we make all of them intro '@property's, and use f.ex.:
# ```
# mock_layer = patch.object(SavableData, "layer", new_callable=PropertyMock)
# mock_layer.assert_called()
# ```
@pytest.mark.slow
@pytest.mark.stub
def test_stubbed_data_onto_plot(window):
    """
    Loads stubbed data into window plottables, then runs it through the dim-reduction process.

    If this fails, it may be because dim-reduction requirements are updated.
    """
    consts.flags["truncate"] = True  # Reduce dataset real quick
    window.data = load_savable()
    window.start_cooking_iii()


@pytest.mark.slow
@pytest.mark.stub
def test_load_through_primary_window(window, obj_fixture):
    """
    Test loading the stubbed save and then plotting it.

    If this test fails, it may be because save requirements are updated.
    """
    # Mock to fix the dialog window
    mocked_qfiledialog = patch.object(
        QFileDialog, "getOpenFileName", return_value=(SAVE_LOCATION, "")
    )
    with mocked_qfiledialog:
        # load_file_wrapper calls utilize_data, which applies the plottables to the plot
        window.load_file_wrapper()
