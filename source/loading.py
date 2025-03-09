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


class FileDialogManager():
    """
    Class to simplify handling of files and file dialogs.

    Methods:
        find_file -- generic function to open a qfiledialog; for wrapping
        find_some_file -- wrapper for find_file which has filters set for all files
        find_picture_file -- wrapper for find_file which has filters set for pictures
        find_trained_model_file -- wrapper for find_file which has filters set for trained nn models
    """

    def __init__(self, parent_window):
        """Initilalizes with a parent window to say to whom any of its spawned window is a child."""
        self.parent = parent_window

    def find_file(self, file_filters=consts.FILE_FILTERS.values() ):
        """
        Launch a file dialog and return the filepath and selected filter.

        Note that the first element in file_filters is used as the initial file filter.
        """
        # Test whether file_filters is a-subset-of/equal-to the legal file filters
        if not set(file_filters) <= set(consts.FILE_FILTERS.values()):
            raise RuntimeError("Unacceptable list of file filters")

        # Generate Qt-readable filter specifications
        file_filters = list(file_filters)
        initial_filter = file_filters[0]
        filters = ";;".join(file_filters)

        # This function opens a nifty Qt-made file dialog
        filepath, selected_filter = QFileDialog.getOpenFileName(
            parent=self.parent,
            filter=filters,
            initialFilter=initial_filter,
        )

        return filepath, selected_filter

    def find_some_file(self):
        """Launch a file dialog where user is prompted to pick out any file their heart desires."""
        path, _ = self.find_file()
        return path

    def find_picture_file(self):
        """Launch file dialog where user is intended to pick out a graphical image file."""
        filters = [
            consts.FILE_FILTERS["pictures"],
            consts.FILE_FILTERS["whatever"],
        ]
        path, _ = self.find_file(filters)
        return path

    def find_trained_model_file(self):
        """Launch file dialog where user is intended to pick out the file for a trained nn-model."""
        filters = [
            consts.FILE_FILTERS["pytorch"],
            consts.FILE_FILTERS["whatever"],
        ]
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


def reduce_data(trained_file, categories, target_dimensionality=2):
    """Take a homogenous array of data, and reduce its dimensionality through t-SNE."""
    # TEMP: This is a hard-coded simulation of choosing a discrete layer
    model_obj = AutoencodeModel(categories, trained_file)
    model_obj.load_from_checkpoint()
    model_obj.to(model_obj.device)
    model_obj.eval()

    features = []

    # @Wilhelmsen: What *is* the output?
    def my_hook(model, args, output):
        features.append(output.detach())

    #def hooker(d, keyname):
    #    def hook(model, args, output):
    #        d[keyname] = output.detach()
    #    return hook

    # Gets the "learnable parameters" from the model's state_dict
    parameters = list(model_obj.state_dict().values())
    # Selects a small slice of the parameters to t-SNE
    selected_features = parameters[163:167]
    selected_features = np.array(selected_features)

    # Else try layer4.0
    # Notice that the layer is here: ~~~~~~vvvvvv
    hook_handle = model_obj.model.backbone.layer4.register_forward_hook(my_hook)

    # Forward the model
    #with torch.no_grad():
    #    _ = model_obj()

    #print("Number of entries in features:", len(features))

    hook_handle.remove()

    # Reduce dimensionality by t-SNE
    perplexity_n = min(30, len(selected_features) - 1)
    np.random.seed(42)  # @Wilhelmsen: Define seed elsewhere, once data has been visualized to graph
    tsne = TSNE(n_components=target_dimensionality, perplexity=perplexity_n)
    reduced_data = tsne.fit_transform(selected_features)

    return reduced_data


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
