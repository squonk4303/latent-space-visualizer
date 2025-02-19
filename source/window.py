#!/usr/bin/env python3
import sys
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QWidget,
)

from consts import Const
# from debug import Color


class StackedLayoutManager():
    """
    Class to handle the layout
    It's like a data structure I made
    """
    def __init__(self, items=None):
        """ Creates an empty stacked layout """
        self.layout = QStackedLayout()
        self.items = list() if items is None else items
        self.selected_item = len(self.items) - 1  # [sic.], -1 on no items

    # TODO: removing items

    def add_widget(self, widget):
        """ Appends a widget to the layout """
        self.layout.addWidget(widget)
        self.items.append(widget)
        self.selected_item += 1

    def add_layout(self, layout):
        """ Appends a layout to the layout """
        tab = QWidget()
        tab_layout = layout
        tab.setLayout(tab_layout)
        self.layout.addWidget(tab)
        self.items.append(self.layout)
        self.selected_item += 1

    def scroll_forth(self, n=1):
        """ Scrolls to next layer """
        maximum = len(self.items)
        self.selected_item = (self.selected_item + n) % maximum
        self.layout.setCurrentIndex(self.selected_item)

    def scroll_back(self, n=1):
        """ Scrolls to next layer """
        maximum = len(self.items)
        self.selected_item = (self.selected_item - n) % maximum
        self.layout.setCurrentIndex(self.selected_item)


class MainWindow(QMainWindow):
    """
    Main window of the program
    This is where all the action happens
    """
    def __init__(self, *args, **kwargs):
        """ Constructor for the main window """
        super().__init__(*args, **kwargs)

        # --- Initiations ---
        self.initiate_menu_bar()

        # --- Layout ---
        # Definitions
        stack_layout = StackedLayoutManager()
        open_file_layout = QHBoxLayout()
        self.open_file_button = QPushButton(Const.OPEN_FILE_LABEL)

        # Layout Organization
        open_file_layout.addWidget(self.open_file_button)
        stack_layout.add_layout(open_file_layout)

        # Widgets
        self.open_file_button.clicked.connect(self.get_filename)
        # TODO: Should this be a qaction?     ^^^^^^^^^^^^^^^^^

        # --- Window Settings ---
        self.resize(650, 450)
        self.setWindowTitle(Const.WINDOW_TITLE)
        widget = QWidget()
        widget.setLayout(stack_layout.layout)
        self.setCentralWidget(widget)

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
