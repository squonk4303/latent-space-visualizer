#!/usr/bin/env python3
import torch
import mmap
import tempfile

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
    print(type(model))
    #print(model)
    return model

def layer_summary(loaded_model):
    input_txt = str(loaded_model)

    #Create a temporary data file to store data in a list
    lines = []
    with tempfile.TemporaryFile("wb+", 0) as file:
       file.write(input_txt.encode("utf-8"))
       mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
       print("\n\n")
       while True:
            byteline = mm.readline()
            if byteline:
                lines.append(byteline.decode("utf-8"))
            else: 
                break
       mm.close()
    
    #Print content of list 
    for i, line in enumerate(lines):
        print(f"{i}: {line}")