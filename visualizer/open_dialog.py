#!/usr/bin/env python3
"""
Module with functions tohandle file dialogs.
"""
import datetime
import os
from PyQt6.QtWidgets import QFileDialog
from visualizer import consts


def for_file(
    # *, yet TODO; @Wilhelmsen: make all except caption kw-only
    parent,
    file_filters=consts.FILE_FILTERS.values(),
    caption="",
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


def for_some_file(parent, caption=""):
    """Launch a file dialog where user is prompted to pick out any file their heart desires."""
    filepath, _ = for_file(parent, caption=caption)
    return filepath


def for_image_file(parent, caption=""):
    """Launch file dialog where user is intended to pick out a graphical image file."""
    filters = [
        consts.FILE_FILTERS["pictures"],
        consts.FILE_FILTERS["whatever"],
    ]
    filepath, _ = for_file(parent, filters, caption=caption)
    return filepath


def for_trained_model_file(parent, caption=""):
    """Launch file dialog where user is intended to pick out the file for a trained nn-model."""
    filters = [
        consts.FILE_FILTERS["pytorch"],
        consts.FILE_FILTERS["whatever"],
    ]
    filepath, _ = for_file(parent, filters, caption=caption)
    return filepath


def for_directory(parent, caption=""):
    """Launch a file dialog where user is prompted to pick out a directory."""
    dirpath = QFileDialog.getExistingDirectory(
        parent=parent,
        caption=caption,
    )
    return dirpath


def to_save_file(
    caption="",
    *,
    parent,
    file_filters=consts.FILE_FILTERS.values(),
    options=QFileDialog.Option.ReadOnly,
):
    """."""
    # Test whether file_filters is a subset of-- or equal to, the legal file filters
    if not set(file_filters) <= set(consts.FILE_FILTERS.values()):
        raise RuntimeError("Unacceptable list of file filters")

    # Generate Qt-readable filter specifications
    # @Wilhelmsen: Find a way to make the initial_filter to be a reasonable
    # choice, instead of random, which it is now. Perhaps make the
    # FILE_FILTERS an ordered dict, and place "whatever" first?
    file_filters = list(file_filters)
    initial_filter = file_filters[0]
    filters = ";;".join(file_filters)

    # Make default filename based on date/time
    default_filename = datetime.datetime.now().strftime("%y%m%d-%H%M%S-data.pickle")

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
