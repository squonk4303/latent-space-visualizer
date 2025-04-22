#!/usr/bin/env python3
"""
Module with functions tohandle file dialogs.
"""
import datetime
from PyQt6.QtWidgets import (
    QFileDialog,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton
    )
from visualizer import consts


def for_file(
    caption="",
    *,
    parent,
    file_filters=consts.FILE_FILTERS.values(),
    options=QFileDialog.Option.ReadOnly,
):
    """
    Launch a file dialog and return the filepath and selected filter.

    Note that the first element in file_filters is used as the initial file filter.
    """
    # Test whether file_filters is a subset of-- or equal to, the legal file filters
    if not set(file_filters) <= set(consts.FILE_FILTERS.values()):
        raise RuntimeError("Unacceptable list of file filters")

    # Generate Qt-readable filter specifications
    file_filters = list(file_filters)
    initial_filter = file_filters[0]
    filters = ";;".join(file_filters)

    # This function opens a nifty Qt-made file dialog
    filepath, selected_filter = QFileDialog.getOpenFileName(
        parent=parent,
        caption=caption,
        filter=filters,
        initialFilter=initial_filter,
        options=options,
    )

    return filepath, selected_filter


def to_save_file(
    caption="",
    *,
    parent,
    file_filters=consts.FILE_FILTERS.values(),
    options=QFileDialog.Option.ReadOnly,
):
    """Launch file dialog where user is prompted for where to save a .pickle file."""
    # Test whether file_filters is a subset of-- or equal to, the legal file filters
    if not set(file_filters) <= set(consts.FILE_FILTERS.values()):
        raise RuntimeError("Unacceptable list of file filters")

    # Generate Qt-readable filter specifications
    file_filters = list(file_filters)
    initial_filter = file_filters[0]
    filters = ";;".join(file_filters)

    # Make default filename based on date/time
    default_filename = datetime.datetime.now().strftime("save_data/%y%m%d-%H%M%S-data.pickle")

    # This function opens a nifty Qt-made file dialog
    filepath, selected_filter = QFileDialog.getSaveFileName(
        parent=parent,
        caption=caption,
        directory=default_filename,
        filter=filters,
        initialFilter=initial_filter,
        options=options,
    )

    return filepath, selected_filter


def for_some_file(caption="", *, parent):
    """Launch a file dialog where user is prompted to pick out any file their heart desires."""
    filepath, _ = for_file(caption, parent=parent)
    return filepath


def for_image_file(caption=consts.IMAGE_DIALOG_CAPTION, *, parent):
    """Launch file dialog where user is intended to pick out a graphical image file."""
    filters = [
        consts.FILE_FILTERS["pictures"],
        consts.FILE_FILTERS["whatever"],
    ]
    filepath, _ = for_file(caption, parent=parent, file_filters=filters)
    return filepath


def for_trained_model_file(caption=consts.TRAINED_MODEL_DIALOG_CAPTION, *, parent):
    """Launch file dialog where user is intended to pick out the file for a trained nn-model."""
    filters = [
        consts.FILE_FILTERS["pytorch"],
        consts.FILE_FILTERS["whatever"],
    ]
    filepath, _ = for_file(caption, parent=parent, file_filters=filters)
    return filepath


def for_directory(caption="", *, parent):
    """Launch a file dialog where user is prompted to pick out a directory."""
    dirpath = QFileDialog.getExistingDirectory(
        parent=parent,
        caption=caption,
    )
    return dirpath

class LayerDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle("Select layer(s)")

        self.startButton = QComboBox(self)
        self.startButton.addItem("...")

        self.endButton = QComboBox(self)
        self.endButton.addItem("...")

        submitButton = QPushButton("Submit")

        layout = QVBoxLayout()
        label = QLabel("Layers go here")
        subLayout = QHBoxLayout()

        subLayout.addWidget(self.startButton)
        subLayout.addWidget(self.endButton)

        layout.addWidget(label)
        layout.addLayout(subLayout)
        layout.addWidget(submitButton)
    
    def expand_buttons(self, layers):

        for layer in layers:
            self.startButton.addItem(layer)
            self.endButton.addItem(layer)
        
def for_layer_select(parent):
    return None
