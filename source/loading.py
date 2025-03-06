#!/usr/bin/env python3
import torch
import mmap
import tempfile
from sklearn.manifold import TSNE
import numpy as np

from PyQt6.QtWidgets import (
    QFileDialog,
)

import consts
import external.fcn as fcn
import utils


class FileDialogManager():
    """
    Class to simplify handling of files and file dialogs.

    Methods:
        __init__ -- initializes with argument being which window is the parent
        find_file -- generic function to open a qfiledialog; for wrapping
        find_some_file -- wrapper for find_file which has filters set for all files
        find_picture_file -- wrapper for find_file which has filters set for pictures
        find_trained_model_file -- wrapper for find_file which has filters set for trained nn models
    """

    def __init__(self, parent_window):
        self.parent = parent_window

    def find_file(self, file_filters=list(consts.FILE_FILTERS.values())):
        """Launch a file dialog and return the filepath and selected filter."""
        if not (utils.arr_is_subset(file_filters, list(consts.FILE_FILTERS.values()))):
            raise RuntimeError("Unacceptable list of file filters")

        initial_filter = file_filters[0]
        filters = ";;".join(file_filters)

        filepath, selected_filter = QFileDialog.getOpenFileName(
            self.parent,
            filter=filters,
            initialFilter=initial_filter,
        )

        return filepath, selected_filter

    def find_some_file(self):
        """Launches file dialog and sets the returned filepath to local attribute."""
        path, _ = self.find_file()
        return path

    def find_picture_file(self):
        """Launches file dialog for pictures and uses return value for attribute."""
        filters = [consts.FILE_FILTERS["pictures"]]
        path, _ = self.find_file(filters)
        return path

    def find_trained_model_file(self):
        """Launches file dialog for nn-models and uses return value for attribute."""
        filters = [consts.FILE_FILTERS["pytorch"]]
        path, _ = self.find_file(filters)
        return path


class AutoencodeModel(fcn.FCNResNet101):
    def __init__(self, cat, path):
        super().__init__(cat)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cats = cat
        self.path = path

    def load_from_checkpoint(self):
        checkpoint = torch.load(self.path, map_location=self.device,
                                weights_only=False)

        state_dict = checkpoint["state_dict"]
        new_state_dict = dict()
        for key, value in state_dict.items():
            new_key = key.removeprefix("module.")
            new_state_dict[new_key] = value

        checkpoint["state_dict"] = new_state_dict

        self.load_state_dict(checkpoint["state_dict"], strict=True)
        return self


def get_model(trained_file, categories):
    """."""
    model_obj = AutoencodeModel(categories, trained_file)
    model_obj = model_obj.load_from_checkpoint()
    model_obj = model_obj.to(model_obj.device)
    model_obj.eval()
    return model_obj


def print_dim_reduced(trained_file, categories=["skin"]):
    """Print a slice of the model, with reduced dimensionality through t-SNE."""
    # Load model from checkpoint to memory
    # And do some set-up, such as move to device
    model_obj = AutoencodeModel(categories, trained_file)
    model_obj = model_obj.load_from_checkpoint()
    model_obj = model_obj.to(model_obj.device)
    model_obj.eval()

    # Get the features from state_dict
    layer_dict = model_obj.state_dict()   # returns an "OrderedDict" object

    features_list = list(layer_dict.values())

    # Choose a layer to represent
    # TODO: Hardcoding this for now
    selected_features = features_list[163:167]
    selected_features = np.array(selected_features)

    # Reduce dimensionality by t-SNE
    perplexity_n = min(30, len(selected_features) - 1)
    np.random.seed(42)  # TODO set seed somewhere better
    tsne = TSNE(n_components=2, perplexity=perplexity_n)
    dim_reduced = tsne.fit_transform(selected_features)

    # Computer. Show me the t-SN Embedded layer
    print(type(dim_reduced))
    print(dim_reduced)


def layer_summary(loaded_model, start_layer=0, end_layer=0):
    """
    Summarises selected layers from a given model objet.
    If endlayer is left blank only return one layer.
    If start layer is left blank returns all layers.
    If both layers are specified returns from startlayer up to
    and including the endlayer!
    """
    # Sets basic logic and variables
    all_layers = False
    if not end_layer:
        end_layer = start_layer
    if not start_layer:
        all_layers = True

    input_txt = str(loaded_model)
    target = "layer"
    # Assigns targetlayers for use in search later
    next_layer = target + str(end_layer+1)
    target += str(start_layer)

    """
    At some point in this function an extraction function is to be added
    to filter the information and only return the useful information and attributes
    to be added to the list. For now it takes the entire line of information.
    """

    # Create a temporary data file to store data in a list
    lines = []
    with tempfile.TemporaryFile("wb+", 0) as file:
        file.write(input_txt.encode("utf-8"))
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        while True:
            byteline = mm.readline()
            if byteline:
                lines.append(byteline.decode("utf-8"))
            else:
                break
        mm.close()

    # Returns selected layers
    found = False
    eol = False
    new = 0
    for i, line in enumerate(lines):
        if all_layers:
            pass
        elif target in line:
            found = True
        elif next_layer in line:
            eol = True
            new = i
        if all_layers or found and not eol:
            print(f"{i}: {line}", end="")

    # End of print
    if all_layers:
        print("\nEOF: no more lines")
    else:
        print(f"\nNext line is {new}: {lines[new]}")
