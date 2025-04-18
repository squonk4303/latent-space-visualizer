#!/usr/bin/env python3
from unittest.mock import patch

import numpy as np
import pytest
import torch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog

from visualizer import consts, utils, loading
from visualizer.models.segmentation import FCNResNet101
from visualizer.plottables import SavableData
from visualizer.view_manager import PrimaryWindow
import visualizer.models.segmentation


# --- Fixtures and Sort-of-Fixtures ---
@pytest.fixture
def primary_window(qtbot):
    """
    Initializes the primary window of the application for use in following tests
    Returns class PrimaryWindow(QMainWindow)
    """
    window = PrimaryWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def data_object():
    data = SavableData()
    data.model = FCNResNet101()
    data.model.load(consts.MULTILABEL_MODEL)
    data.layer = "layer4"
    data.paths = utils.grab_image_paths_in_dir(consts.S_DATASET)
    data.dataset_location = consts.S_DATASET
    data.dataset_intermediary = torch.rand(1, 3, 640, 640)
    return data


@pytest.fixture
def valid_model():
    model = FCNResNet101()
    model.load(consts.MULTILABEL_MODEL)
    return model


mocked_trained_model_qfiledialog = patch.object(
    QFileDialog,
    "getOpenFileName",
    return_value=(consts.MULTILABEL_MODEL, consts.FILE_FILTERS["whatever"]),
)


mocked_cancelled_qfiledialog = patch.object(
    QFileDialog, "getOpenFileName", return_value=("", consts.FILE_FILTERS["whatever"])
)


def test_window_basically(primary_window, qtbot):
    """Test that the window is alive and exists."""
    assert primary_window.windowTitle() == consts.STAGE_TITLE
    assert primary_window.centralWidget()


# @Wilhelmsen: Reconsider this
def _test_qaction_to_switch_tabs(primary_window, qtbot):
    """Test switching tabs with the QActions."""
    assert primary_window.tab_layout.currentIndex() == 0
    primary_window.next_tab.trigger()
    assert primary_window.tab_layout.currentIndex() == 1
    primary_window.next_tab.trigger()
    assert primary_window.tab_layout.currentIndex() == 0
    primary_window.prev_tab.trigger()
    assert primary_window.tab_layout.currentIndex() == 1


# @Wilhelmsen: Reconsider this
def _test_buttons_to_switch_tabs(primary_window, qtbot):
    """Tests switching tabs with hardcoded buttons"""
    qtbot.mouseClick(primary_window.start_tab_button, Qt.MouseButton.LeftButton)
    assert primary_window.tab_layout.currentIndex() == 0
    qtbot.mouseClick(primary_window.graph_tab_button, Qt.MouseButton.LeftButton)
    assert primary_window.tab_layout.currentIndex() == 1


@pytest.mark.require_pretrained_model
@pytest.mark.slow
def test_tab_switch_after_selecting_file(primary_window, qtbot):
    """
    Test switching to next tab after selecting file.
    Currently a bit slow, for it loads and evaluates a pretrained model...
    """
    with mocked_trained_model_qfiledialog:
        assert primary_window.tab_layout.currentIndex() == 0
        primary_window.action_to_open_file.trigger()
        assert primary_window.tab_layout.currentIndex() == 0


def test_cancelled_file_select(primary_window, qtbot):
    """
    Test that a cancelled file dialog exits gracefully.

    Assertions are for whether the tab changes, but on failure to exit the
    dialog, the program would throw an error, so that's being tested here
    as well.
    """
    with mocked_cancelled_qfiledialog:
        assert primary_window.tab_layout.currentIndex() == 0
        primary_window.action_to_open_file.trigger()
        assert primary_window.tab_layout.currentIndex() == 0


@pytest.mark.slow
def test_quicksave_n_quickload(primary_window, data_object):
    """
    Quicksave and quickload an object to assert if the loaded copy is equal.

    NOTE: Overwrites the quicksave.
    """
    primary_window.data = data_object
    primary_window.quicksave_action.trigger()
    primary_window.quickload_action.trigger()
    assert np.array_equal(
        primary_window.data.dataset_intermediary, data_object.dataset_intermediary
    )
    # Assert that .data and data_object don't point to the same object
    assert primary_window.data is not data_object
    primary_window.data.dataset_intermediary = torch.rand(1, 3, 640, 640)
    assert not np.array_equal(
        primary_window.data.dataset_intermediary, data_object.dataset_intermediary
    )


@pytest.mark.slow
@pytest.mark.stub
def test_try_to_activate_goforit_button(primary_window, valid_model):
    """
    Assert that button starts disabled, and then is enabled when all conditions are fulfilled.
    """
    primary_window.try_to_activate_goforit_button()
    assert not primary_window.go_for_it_button.isEnabled()
    primary_window.data.model = valid_model
    primary_window.data.layer = "layer4"
    primary_window.data.dataset_location = consts.MEDIUM_DATASET
    primary_window.try_to_activate_goforit_button()
    assert primary_window.go_for_it_button.isEnabled()


def _test_find_layer_activates_goforit_button(primary_window, valid_model):
    # @Linnea: Update this when we have a proper findlayer function
    # Mock to assure the function sets a valid layer
    mocked_findlayer = patch.object()
    with mocked_findlayer:
        # Assert the function doesn't enable the button erroneously
        primary_window.find_layer()
        assert not primary_window.go_for_it_button.isEnabled()
        primary_window.data.model = valid_model
        primary_window.data.dataset_location = consts.MEDIUM_DATASET
        # Assert the final function changes the button state
        primary_window.find_layer()
        assert primary_window.go_for_it_button.isEnabled()


@pytest.mark.slow
@pytest.mark.stub
def test_find_model_activates_goforit_button(primary_window):
    # Mock to assure the function will set a valid trained model
    with mocked_trained_model_qfiledialog:
        # Assert the function doesn't enable the button erroneously
        primary_window.load_model_file()
        assert not primary_window.go_for_it_button.isEnabled()
        primary_window.data.layer = "layer4"
        primary_window.data.dataset_location = consts.MEDIUM_DATASET
        primary_window.data.model = FCNResNet101()
        # Assert the final function changes the button state
        primary_window.load_model_file()
        assert primary_window.go_for_it_button.isEnabled()


def test_select_dim_reduction_activates_goforit_button(primary_window):
    return
    primary_window.suggest_model_type("...")
    assert not primary_window.go_for_it_button.isEnabled()
    primary_window.data.layer = "layer4"
    primary_window.data.dataset_location = consts.MEDIUM_DATASET

    # Assert the final function changes the button state
    primary_window.suggest_model_type(const.MODEL_TYPES[0])


@pytest.mark.slow
@pytest.mark.stub
def test_find_dataset_activates_goforit_button(primary_window, valid_model):
    # Mock to assure the function will set a valid dataset
    mocked_directory_dialog = patch.object(
        QFileDialog, "getExistingDirectory", return_value=consts.MEDIUM_DATASET
    )
    with mocked_directory_dialog:
        # Assert the function doesn't enable the button erroneously
        assert not primary_window.go_for_it_button.isEnabled()
        primary_window.data.model = valid_model
        primary_window.data.layer = "layer4"
        # Assert the final function changes the button state
        primary_window.find_dataset()
        assert primary_window.go_for_it_button.isEnabled()


def test_automatic_getting_of_model_types(primary_window):
    """Assert that all model types in the const are valid."""
    for model in consts.MODEL_TYPES:
        primary_window.suggest_model_type(model)


def test_suggest_model_type_dont_take_no_shit(primary_window):
    """Make sure that the function *does* raise an error on bad model type."""
    with pytest.raises(ValueError):
        primary_window.suggest_model_type("Bogus model for fools and knaves")


def test_automatic_getting_of_dim_reductions(primary_window):
    """
    Assert that all model types in the const are valid.
    NOTE: Doesn't actyally use a const yet. @Wilhelmsen.
    """
    dim_reduction_techs = {
        "TSNE": print,
        "PCA": print,
        "UMAP": print,
        "TRIMAP": print,
        "PACMAP": print,
    }
    for technique in dim_reduction_techs:
        primary_window.suggest_dim_reduction(technique)


def test_suggest_dim_reduction_dont_take_no_shit(primary_window):
    """Make sure that the function *does* raise an error on bad technique."""
    with pytest.raises(ValueError):
        primary_window.suggest_model_type("Bogus technique for fools and knaves")


def _test_dim_techniques_from_dict(primary_window):
    """@Wilhelmsen: doesn't work right now. Try agains later."""
    from visualizer.view_manager import dim_reduction_techs
    from visualizer.loading import tsne

    mocked_tsne = patch(
        "loading.tsne", side_effect=SystemExit("mocked_tsne called; stopping program")
    )
    primary_window.data.dim_reduction = "TSNE"
    arr = torch.rand(3, 28, 28)
    with mocked_tsne:
        dim_reduction_techs[primary_window.data.dim_reduction](arr)
