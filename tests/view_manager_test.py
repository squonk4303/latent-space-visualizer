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
def window(qtbot):
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
    data.layer = consts.LAYER
    data.paths = utils.grab_image_paths_in_dir(consts.S_DATASET)
    data.dataset_location = consts.S_DATASET
    data.dataset_intermediary = torch.rand(1, 3, 640, 640)
    return data


mocked_trained_model_qfiledialog = patch.object(
    QFileDialog,
    "getOpenFileName",
    return_value=(consts.MULTILABEL_MODEL, consts.FILE_FILTERS["whatever"]),
)


mocked_cancelled_qfiledialog = patch.object(
    QFileDialog, "getOpenFileName", return_value=("", consts.FILE_FILTERS["whatever"])
)


def test_window_basically(window, qtbot):
    """Test that the window is alive and exists."""
    assert window.windowTitle() == consts.STAGE_TITLE
    assert window.centralWidget()


# @Wilhelmsen: Reconsider this
def _test_qaction_to_switch_tabs(window, qtbot):
    """Test switching tabs with the QActions."""
    assert window.tab_layout.currentIndex() == 0
    window.next_tab.trigger()
    assert window.tab_layout.currentIndex() == 1
    window.next_tab.trigger()
    assert window.tab_layout.currentIndex() == 0
    window.prev_tab.trigger()
    assert window.tab_layout.currentIndex() == 1


# @Wilhelmsen: Reconsider this
def _test_buttons_to_switch_tabs(window, qtbot):
    """Tests switching tabs with hardcoded buttons"""
    qtbot.mouseClick(window.start_tab_button, Qt.MouseButton.LeftButton)
    assert window.tab_layout.currentIndex() == 0
    qtbot.mouseClick(window.graph_tab_button, Qt.MouseButton.LeftButton)
    assert window.tab_layout.currentIndex() == 1


@pytest.mark.require_pretrained_model
@pytest.mark.slow
def test_tab_switch_after_selecting_file(window, qtbot):
    """
    Test switching to next tab after selecting file.
    Currently a bit slow, for it loads and evaluates a pretrained model...
    """
    with mocked_trained_model_qfiledialog:
        assert window.tab_layout.currentIndex() == 0
        window.action_to_open_file.trigger()
        assert window.tab_layout.currentIndex() == 0


def test_cancelled_file_select(window, qtbot):
    """
    Test that a cancelled file dialog exits gracefully.

    Assertions are for whether the tab changes, but on failure to exit the
    dialog, the program would throw an error, so that's being tested here
    as well.
    """
    with mocked_cancelled_qfiledialog:
        assert window.tab_layout.currentIndex() == 0
        window.action_to_open_file.trigger()
        assert window.tab_layout.currentIndex() == 0


@pytest.mark.filterwarnings("ignore:No artists")
@pytest.mark.require_pretrained_model
@pytest.mark.slow
def test_quicksave_n_quickload(window, data_object):
    """
    Quicksave and quickload an object to assert if the loaded copy is equal.

    @Wilhelmsen: the data_object this uses requires pretrained model,
                 but that needn't be the case for the test.
                 Change that.
    NOTE: Overwrites the quicksave.
    """
    window.data = data_object
    window.quicksave_action.trigger()
    window.quickload_action.trigger()
    assert np.array_equal(
        window.data.dataset_intermediary, data_object.dataset_intermediary
    )
    # Assert that .data and data_object don't point to the same object
    assert window.data is not data_object
    window.data.dataset_intermediary = torch.rand(1, 3, 640, 640)
    assert not np.array_equal(
        window.data.dataset_intermediary, data_object.dataset_intermediary
    )


@pytest.mark.stub
def test_launch_button_activates_on_layer(window):
    # @Linnea: Update this when we have a proper findlayer function
    # Mock to assure the function sets a valid layer
    # Assert the function doesn't enable the button erroneously
    window.find_layer()
    assert not window.launch_button.isEnabled()
    window.data.dataset_location = consts.MEDIUM_DATASET
    window.data.dim_reduction = "TSNE"
    window.data.model = FCNResNet101()
    window.data.model_location = consts.MULTILABEL_MODEL
    # Assert the final function changes the button state
    window.find_layer()
    assert window.launch_button.isEnabled()


@pytest.mark.stub
def test_launch_button_activates_on_model_location(window):
    # Mock to assure the function will set a valid trained model
    with mocked_trained_model_qfiledialog:
        # Assert the function doesn't enable the button erroneously
        window.load_model_location()
        assert not window.launch_button.isEnabled()
        window.data.dataset_location = consts.MEDIUM_DATASET
        window.data.dim_reduction = "TSNE"
        window.data.layer = consts.LAYER
        window.data.model = FCNResNet101()
        # Assert the final function changes the button state
        window.load_model_location()
        assert window.launch_button.isEnabled()


@pytest.mark.stub
def test_launch_button_activates_on_model_type(window):
    # Assert the function doesn't enable the button erroneously
    window.suggest_model_type(consts.NIL)
    window.suggest_model_type(consts.MODEL_TYPES[0])
    assert not window.launch_button.isEnabled()

    window.data.dataset_location = consts.MEDIUM_DATASET
    window.data.dim_reduction = "TSNE"
    window.data.layer = consts.LAYER
    window.data.model_location = consts.MULTILABEL_MODEL
    # Assert the final function changes the button state
    window.suggest_model_type(consts.MODEL_TYPES[0])
    assert window.launch_button.isEnabled()


@pytest.mark.stub
def test_launch_button_activates_on_dim_reduction(window):
    # Assert the function doesn't enable the button erroneously
    window.suggest_dim_reduction("TSNE")
    assert not window.launch_button.isEnabled()

    window.data.dataset_location = consts.MEDIUM_DATASET
    window.data.layer = consts.LAYER
    window.data.model = FCNResNet101()
    window.data.model_location = consts.MULTILABEL_MODEL
    # Assert the final function changes the button state
    window.suggest_dim_reduction("TSNE")
    assert window.launch_button.isEnabled()


@pytest.mark.stub
def test_launch_button_activates_on_dataset(window):
    # Mock to assure the function will set a valid dataset
    mocked_directory_dialog = patch.object(
        QFileDialog, "getExistingDirectory", return_value=consts.MEDIUM_DATASET
    )
    with mocked_directory_dialog:
        # Assert the function doesn't enable the button erroneously
        window.find_dataset()
        assert not window.launch_button.isEnabled()
        window.data.dim_reduction = "TSNE"
        window.data.layer = consts.LAYER
        window.data.model = FCNResNet101()
        window.data.model_location = consts.MULTILABEL_MODEL
        # Assert the final function changes the button state
        window.find_dataset()
        assert window.launch_button.isEnabled()


def test_automatic_getting_of_model_types(window):
    """Assert that all model types in the const are valid."""
    for model in consts.MODEL_TYPES:
        window.suggest_model_type(model)


def test_suggest_model_type_dont_take_no_shit(window):
    """Make sure that the function *does* raise an error on bad model type."""
    with pytest.raises(ValueError):
        window.suggest_model_type("Bogus model for fools and knaves")


def test_automatic_getting_of_dim_reductions(window):
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
        window.suggest_dim_reduction(technique)


def test_suggest_dim_reduction_dont_take_no_shit(window):
    """Make sure that the function *does* raise an error on bad technique."""
    with pytest.raises(ValueError):
        window.suggest_model_type("Bogus technique for fools and knaves")


def _test_dim_techniques_from_dict(window):
    """@Wilhelmsen: doesn't work right now. Try agains later."""
    from visualizer.view_manager import dim_reduction_techs
    from visualizer.loading import tsne

    mocked_tsne = patch(
        "loading.tsne", side_effect=SystemExit("mocked_tsne called; stopping program")
    )
    window.data.dim_reduction = "TSNE"
    arr = torch.rand(3, 28, 28)
    with mocked_tsne:
        dim_reduction_techs[window.data.dim_reduction](arr)
