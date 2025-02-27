import consts

from PyQt6.QtWidgets import (
    QFileDialog
)

class File():
    path = "none"

    def open_dialogue(self, parent):
        initial_filter = consts.FILE_FILTERS[0]
        filters = ";;".join(consts.FILE_FILTERS)

        # TODO: Consider QFileDialog: {FileMode, Acceptmode, "Options"}
        filepath, selected_filter = QFileDialog.getOpenFileName(
            parent,
            filter=filters,
            initialFilter=initial_filter,
        )
        self.path = filepath