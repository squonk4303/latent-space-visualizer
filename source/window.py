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

    def add_widget(self, widget):
        """ Appends a widget to the layout """
        self.layout.addWidget(widget)
        self.items.append(widget)

    def add_layout(self, qlayout):
        """ Appends a layout to the layout """
        tab = QWidget()
        tab.setLayout(qlayout)
        self.layout.addWidget(tab)
        self.items.append(self.layout)

    def scroll_somewhere(self, n=1):
        """ Scrolls to a layer relatively , according to n """
        maximum = len(self.items)
        current_index = self.layout.currentIndex()
        new_index = (current_index + int(n)) % maximum
        self.layout.setCurrentIndex(new_index)

    def scroll_forth(self):
        """ Scrolls to next layer """
        self.scroll_somewhere(1)

    def scroll_back(self):
        """ Scrolls to prev layer """
        self.scroll_somewhere(-1)


class MainWindow(QMainWindow):
    """
    Main window of the program
    This is where all the action happens
    """
    def __init__(self, *args, **kwargs):
        """ Constructor for the main window """
        super().__init__(*args, **kwargs)

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

        # --- Initiations
        self.initiate_menu_bar()

        # --- Signals
        self.TEMP_button.clicked.connect(self.tab_layout.scroll_forth)
        self.openfile_button.clicked.connect(self.get_filename)

        self.empty_tab_button.clicked.connect(self.activate_tab_0)
        self.graph_tab_button.clicked.connect(self.activate_tab_1)

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
        menubar = self.menuBar()

        # Open file action
        action_to_open_file = QAction(consts.OPEN_FILE_LABEL, self)
        action_to_open_file.setStatusTip(consts.STATUS_TIP_TEMP)
        action_to_open_file.setShortcut(QKeySequence("Ctrl+O"))
        action_to_open_file.triggered.connect(self.get_filename)
        # Note --->function Raference         ^^^^^^^^^^^^^^^^^

        next_tab = QAction("TEMP: &Next tab")
        next_tab.setStatusTip(consts.STATUS_TIP_TEMP)
        prev_tab = QAction("TEMP: &Previous tab")
        prev_tab.setStatusTip(consts.STATUS_TIP_TEMP)

        next_tab.triggered.connect(self.tab_layout.scroll_forth)
        prev_tab.triggered.connect(self.tab_layout.scroll_back)

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(action_to_open_file)

        self.navigate_menu = menubar.addMenu("&Tab")
        self.navigate_menu.addAction(next_tab)
        self.navigate_menu.addAction(prev_tab)

        # Export these as "public attributes"
        self.action_to_open_file = action_to_open_file
        self.next_tab = next_tab
        self.prev_tab = prev_tab
        self.file_menu = file_menu

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
