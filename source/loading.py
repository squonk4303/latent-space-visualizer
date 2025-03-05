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
    img_path   = ""
    model_path = ""
    some_path  = ""

    def open_dialog(
            self,
            parent=None,
            file_filter_list=list(consts.FILE_FILTERS.values())
            ):
        """Launch a file dialog and return the filepath and selected filter."""
        if not (utils.arr_is_subset(file_filter_list,
                                    list(consts.FILE_FILTERS.values() ))):
            raise RuntimeError("Unacceptable list of file filters")

        initial_filter = file_filter_list[0]
        filters = ";;".join(file_filter_list)

        filepath, selected_filter = QFileDialog.getOpenFileName(
            parent,
            filter=filters,
            initialFilter=initial_filter,
        )
        if filepath == "":
            # TODO: Make it so the user knows the program stopped
            # because they didn't select a file
            pass
        return filepath, selected_filter

    def open_all(self, parent=None):
        """
        Launches file dialog and sets the returned filepath to local attribute.
        """
        self.some_path, _ = self.open_dialog(parent)

    def open_img(self, parent=None):
        """
        Launches file dialog for pictures and uses return value for attribute.
        """
        filters = [consts.FILE_FILTERS["pictures"]]
        assert filters == [("Image Files (*.png *.jpg *.jpeg *.webp "
                            "*.bmp *.gif *.tif *.tiff *.svg)")]
        self.img_path, _ = self.open_dialog(parent, filters)

    def open_model(self, parent=None):
        """
        Launches file dialog for nn-models and uses return value for attribute.
        """
        filters = [consts.FILE_FILTERS["pytorch"]]
        assert filters == ["PyTorch Files (*.pt *.pth)"]
        self.model_path, _ = self.open_dialog(parent, filters)


class AutoencodeModel(fcn.FCNResNet101):
    def __init__(self, cat, path):
        super().__init__(cat)
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                          else "cpu")
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


def load_method_1(path, list=["skin"]):
    model_class = AutoencodeModel(list, path)
    model = model_class.load_from_checkpoint()
    model = model.to(model_class.device)
    model.eval()
    return model


def _load_method_1(path, arr=["skin"]):
    model_class = AutoencodeModel(arr, path)
    model = model_class.load_from_checkpoint()  # NEED to load from checkpoint
    model = model.to(model_class.device)
    model.eval()  # TODO: at surface seems to do nothing...?
                  # It could be a just-in-case thing...
                  # .eval() makes certain NN-functions act differently I think
    my_state_dict = model_class.state_dict()
    #for i in my_state_dict:
    #    print("-"*64)
    #    print(i)
    #print("-"*64)
    #print(my_state_dict)
    #print("-"*64)

    #print(my_state_dict)
    #features = np.array(list(my_state_dict.values()))

    #print(type(features))
    #print(features)

    #t_sne = TSNE(n_components=2)
    #print(t_sne)
    #t_sne = t_sne.fit_transform(features)
    #print(type(t_sne))
    #print(t_sne)

    #return model


def print_dim_reduced(trained_file, categories=["skin"]):
    # Load model from checkpoint to memory
    # And do some set-up, such as move to device
    model_obj = AutoencodeModel(categories, trained_file)
    model_obj = model_obj.load_from_checkpoint()
    model_obj = model_obj.to(model_obj.device)
    model_obj.eval()

    # Get the features from state_dict
    layer_dict = model_obj.state_dict()   # returns an "OrderedDict" object
    all_features = layer_dict.items()

    # vv~~~  Just prints out to terminal right now
    # for key, value in all_features:
    #     print(key, value.shape)

    features_list = list(layer_dict.values())

    # Choose a layer to represent
    # TODO: Hardcoding this for now
    selected_features = features_list[163:167]
    selected_features = np.array(selected_features)

    # Reduce dimensionality by t-SNE
    perplexity_n = min(30, len(selected_features)-1)
    np.random.seed(42)  # TODO set seed somewhere better
    tsne = TSNE(n_components=2, perplexity=perplexity_n)
    dim_reduced = tsne.fit_transform(selected_features)

    # Computer. Show me the t-SN Embedded layer
    print(type(dim_reduced))
    print(dim_reduced)


def print_state_dict(trained_file, categories):
    # Load model from checkpoint to memory
    # And do some set-up, such as move to device
    m_obj = AutoencodeModel(categories, trained_file)
    m_obj = m_obj.load_from_checkpoint()
    m_obj = m_obj.to(m_obj.device)
    m_obj.eval()

    # Get the features from its state_dict
    dict_obj = m_obj.state_dict()   # returns an "OrderedDict" object
    features = list(dict_obj.values())  # This is now a list of tensors of WAY different dimensionalities

    print(dict_obj)

def layer_summary(loaded_model, start_layer=0, end_layer=0):
    """
    Summarises selected layers from a given model objet.
    If endlayer is left blank only return one layer.
    If start layer is left blank returns all layers.
    If both layers are specified returns from startlayer up to
    and including the endlayer!
    """
    #Sets basic logic and variables
    all_layers = False
    if not end_layer:
        end_layer = start_layer
    if not start_layer:
        all_layers = True

    input_txt = str(loaded_model)
    target = "layer"
    #Assigns targetlayers for use in search later
    next_layer = target + str(end_layer+1)
    target += str(start_layer)


    """
    At some point in this function an extraction function is to be added
    to filter the information and only return the useful information and attributes
    to be added to the list. For now it takes the entire line of information.
    """

    #Create a temporary data file to store data in a list
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

    #Returns selected layers
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
            print(f"{i}: {line}")

    #End of print
    if all_layers:
        print(f"\nEOF: no more lines")
    else:
        print(f"\nNext line is {new}: {lines[new]}")
