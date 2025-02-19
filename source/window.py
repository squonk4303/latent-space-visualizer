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
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

import consts


class StackedLayoutManager():
    """
    Class to handle the layout
    It's like a data structure I made
    TODO: removing items
    """
    def __init__(self, items=None):
        """ Creates an empty stacked layout """
        self.layout = QStackedLayout()
        self.items = list() if items is None else items
        self.selected_item = len(self.items) - 1       # [sic.], -1 on no items

    def add_widget(self, widget):
        """ Appends a widget to the layout """
        self.layout.addWidget(widget)
        self.items.append(widget)
        self.selected_item += 1

    def add_layout(self, qlayout):
        """ Appends a layout to the layout """
        tab = QWidget()
        tab.setLayout(qlayout)
        self.layout.addWidget(tab)
        self.items.append(self.layout)
        self.selected_item += 1

    def scroll_forth(self, n=1):
        """ Scrolls to next layer """
        maximum = len(self.items)
        self.selected_item = (self.selected_item + n) % maximum
        print("Setting", self.selected_item)
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
        # --- Definitions
        greater_layout = QVBoxLayout()
        self.tab_layout = StackedLayoutManager()

        tab_buttons_layout = QHBoxLayout()
        empty_tab = QVBoxLayout()
        graph_tab = QHBoxLayout()

        self.openfile_button = QPushButton(consts.OPEN_FILE_LABEL)
        self.TEMP_button = QPushButton("-->")

        self.empty_tab_button = QPushButton("0")
        self.graph_tab_button = QPushButton("1")

        next_tab = QAction("TEMP: &Next tab")
        next_tab.setStatusTip(consts.STATUS_TIP_TEMP)

        # --- Signals
        self.TEMP_button.clicked.connect(self.tab_layout.scroll_forth)
        self.openfile_button.clicked.connect(self.get_filename)

        self.empty_tab_button.clicked.connect(self.activate_tab_0)
        self.graph_tab_button.clicked.connect(self.activate_tab_1)

        next_tab.triggered.connect(self.tab_layout.scroll_forth)

        # --- Layout Organization
        greater_layout.addLayout(tab_buttons_layout)
        greater_layout.addLayout(self.tab_layout.layout)

        tab_buttons_layout.addWidget(self.empty_tab_button)
        tab_buttons_layout.addWidget(self.graph_tab_button)

        empty_tab.addWidget(self.openfile_button)
        empty_tab.addWidget(self.TEMP_button)

        self.TEMP_label = QLabel("Hi welcome to the graph tab :3")
        graph_tab.addWidget(self.TEMP_label)

        self.tab_layout.add_layout(empty_tab)
        self.tab_layout.add_layout(graph_tab)

        # --- Window Configuration ---
        self.resize(650, 450)
        self.setWindowTitle(consts.WINDOW_TITLE)
        widget = QWidget()
        widget.setLayout(greater_layout)
        self.setCentralWidget(widget)

    def initiate_menu_bar(self):
        menu = self.menuBar()
        self.action_to_open_file = QAction(consts.OPEN_FILE_LABEL, self)
        self.action_to_open_file.setStatusTip(consts.STATUS_TIP_TEMP)
        self.action_to_open_file.setShortcut(QKeySequence("Ctrl+O"))
        self.action_to_open_file.triggered.connect(self.get_filename)
        # Note --->function Raference              ^^^^^^^^^^^^^^^^^

        self.file_menu = menu.addMenu("&File")
        self.file_menu.addAction(self.action_to_open_file)

    def get_filename(self):
        initial_filter = consts.FILE_FILTERS[0]
        filters = ";;".join(consts.FILE_FILTERS)

        # TODO: Consider QFileDialog: {FileMode, Acceptmode, "Options"}
        filename, selected_filter = QFileDialog.getOpenFileName(
            self,
            filter=filters,
            initialFilter=initial_filter,
        )
        print("Result:", filename, selected_filter)

    def activate_tab_0(self):
        self.tab_layout.layout.setCurrentIndex(0)

    def activate_tab_1(self):
        self.tab_layout.layout.setCurrentIndex(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
