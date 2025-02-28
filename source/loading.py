import consts
import external.fcn as fcn
import torch

from PyQt6.QtWidgets import (
    QFileDialog
)
import utils

class File():
    model_path = ""
    img_path = ""
    path = ""

    def open_dialogue(self, parent=None, file_arr=consts.FILE_FILTERS):
        if not (utils.arr_is_subset(file_arr,consts.FILE_FILTERS)):
            print("Error unacceptable file type.")
        else:
            initial_filter = file_arr[0]
            filters = ";;".join(file_arr)

            # TODO: Consider QFileDialog: {FileMode, Acceptmode, "Options"}
            filepath, selected_filter = QFileDialog.getOpenFileName(
                parent,
                filter=filters,
                initialFilter=initial_filter,
            )
            return filepath
        
    def open_all(self, parent=None):
        self.path = self.open_dialogue(parent)
    
    def open_model(self, parent=None):
        self.model_path = self.open_dialogue(parent, [consts.FILE_FILTERS[1]])

    def open_img(self, parent=None):
        self.img_path = self.open_dialogue(parent, [consts.FILE_FILTERS[2]])

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

def load_method_1(path, arr=["skin"]):
    model_class = AutoencodeModel(arr,path)
    model = model_class.load_from_checkpoint()
    model = model.to(model.device)
    model.eval()
    print(type(model))
    print(model)
    return model