#!/usr/bin/env python3
import torch

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

    def open_dialogue(self, parent=None, file_arr=consts.FILE_FILTERS):
        # TODO: Prepare for cases where this is cancelled and for when a bad file is selected
        if not (utils.arr_is_subset(file_arr,consts.FILE_FILTERS)):
            raise RuntimeError("Unacceptable list of file filters")
        else:
            initial_filter = file_arr[0]
            filters = ";;".join(file_arr)

            filepath, selected_filter = QFileDialog.getOpenFileName(
                parent,
                filter=filters,
                initialFilter=initial_filter,
            )
            return filepath, selected_filter

    def open_all(self, parent=None):
        self.some_path, _ = self.open_dialogue(parent)

    def open_img(self, parent=None):
        filters = [consts.FILE_FILTERS[2]]
        self.img_path, _ = self.open_dialogue(parent, filters)

    def open_model(self, parent=None):
        filters = [consts.FILE_FILTERS[1]]
        self.model_path, _ = self.open_dialogue(parent, filters)


class AutoencodeModel(fcn.FCNResNet101):
    def __init__(self, cat, path):
        super().__init__(cat)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cats = cat
        self.path = path

    def load_from_checkpoint(self):
        checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)

        state_dict = checkpoint["state_dict"]
        new_state_dict = dict()
        for key, value in state_dict.items():
            new_key = key.removeprefix("module.")
            new_state_dict[new_key] = value

        checkpoint["state_dict"] = new_state_dict

        self.load_state_dict(checkpoint["state_dict"], strict=True)
        return self

def load_method_1(path, arr=["skin"]):
    model_class = AutoencodeModel(arr,path)
    model = model_class.load_from_checkpoint()
    model = model.to(model_class.device)
    model.eval()
    print(type(model))
    print(model)
    return model
