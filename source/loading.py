#!/usr/bin/env python3
import torch
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

    def open_dialogue(
            self,
            parent=None,
            file_arr=list(consts.FILE_FILTERS.values())
            ):
        """Launch a file dialog and return the filepath and selected filter."""
        if not (utils.arr_is_subset(file_arr,
                                    list(consts.FILE_FILTERS.values() ))):
            raise RuntimeError("Unacceptable list of file filters")

        initial_filter = file_arr[0]
        filters = ";;".join(file_arr)

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
        self.some_path, _ = self.open_dialogue(parent)

    def open_img(self, parent=None):
        """
        Launches file dialog for pictures and uses return value for attribute.
        """
        filters = [consts.FILE_FILTERS["pictures"]]
        assert filters == [("Image Files (*.png *.jpg *.jpeg *.webp "
                            "*.bmp *.gif *.tif *.tiff *.svg)")]
        self.img_path, _ = self.open_dialogue(parent, filters)

    def open_model(self, parent=None):
        """
        Launches file dialog for nn-models and uses return value for attribute.
        """
        filters = [consts.FILE_FILTERS["pytorch"]]
        assert filters == ["PyTorch Files (*.pt *.pth)"]
        self.model_path, _ = self.open_dialogue(parent, filters)


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

    print(my_state_dict)
    features = np.array(list(my_state_dict.values()))

    print(type(features))
    print(features)

    t_sne = TSNE(n_components=2)
    print(t_sne)
    t_sne = t_sne.fit_transform(features)
    print(type(t_sne))
    print(t_sne)

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
