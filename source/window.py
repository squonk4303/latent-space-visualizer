#!/usr/bin/env python3
import sys
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QPushButton,
    QStackedLayout,
)

from consts import Const
# from debug import Color


class StackedLayoutManager():
    """
    Class to handle the layout
    It's like a data structure I made
    """
    def __init__(self, layers=None):
        """ Creates an empty stacked layout """
        self.structure = QStackedLayout()
        self.layers = list() if layers is None else layers
        self.selected_layer = -1 if layers is None else len(layers) - 1

    # TODO: removing widgets

    def add_widget(self, widget):
        """ Appends a widget to the layout """
        self.structure.addWidget(widget)
        self.layers.append(widget)
        self.selected_layer += 1

    def scroll_forth(self, n=1):
        """ Scrolls to next layer """
        maximum = len(self.layers)
        self.selected_layer = (self.selected_layer + n) % maximum
        self.structure.setCurrentIndex(self.selected_layer)

    def scroll_back(self, n=1):
        """ Scrolls to next layer """
        maximum = len(self.layers)
        self.selected_layer = (self.selected_layer - n) % maximum
        self.structure.setCurrentIndex(self.selected_layer)


class MainWindow(QMainWindow):
    """
    Main window of the program
    This is where all the action happens
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor for the main window
        """
        super().__init__(*args, **kwargs)

        self.initiate_menu_bar()

        self.open_file_button = QPushButton(Const.OPEN_FILE_LABEL)
        self.open_file_button.clicked.connect(self.get_filename)
        # TODO: Should this be a qaction?     ^^^^^^^^^^^^^^^^^

        self.setWindowTitle(Const.WINDOW_TITLE)
        self.setCentralWidget(self.open_file_button)

    def initiate_menu_bar(self):
        menu = self.menuBar()
        self.action_to_open_file = QAction(Const.OPEN_FILE_LABEL, self)
        self.action_to_open_file.setStatusTip(Const.STATUS_TIP_TEMP)
        self.action_to_open_file.setShortcut(QKeySequence("Ctrl+O"))
        self.action_to_open_file.triggered.connect(self.get_filename)
        # Note the function Raference              ^^^^^^^^^^^^^^^^^

        self.file_menu = menu.addMenu("&File")
        self.file_menu.addAction(self.action_to_open_file)

    def get_filename(self):
        initial_filter = Const.FILE_FILTERS[0]
        filters = ";;".join(Const.FILE_FILTERS)

        # TODO: Consider QFileDialog: {FileMode, Acceptmode, "Options"}
        filename, selected_filter = QFileDialog.getOpenFileName(
            self,
            filter=filters,
            initialFilter=initial_filter,
        )
        print("Result:", filename, selected_filter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
