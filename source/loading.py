import consts

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

    
        