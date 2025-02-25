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
    QVBoxLayout,
    QWidget,
)

import consts
from stacked_layout_manager import StackedLayoutManager
from plot_widget import PlotWidget


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
        self.openfile_button.clicked.connect(self.open_file_dialog)

        self.empty_tab_button.clicked.connect(self.activate_tab_0)
        self.graph_tab_button.clicked.connect(self.activate_tab_1)

        # --- Layout Organization
        greater_layout.addLayout(tab_buttons_layout)
        greater_layout.addLayout(self.tab_layout)

        tab_buttons_layout.addWidget(self.empty_tab_button)
        tab_buttons_layout.addWidget(self.graph_tab_button)

        empty_tab.addWidget(self.openfile_button)
        empty_tab.addWidget(self.TEMP_button)

        self.plot = PlotWidget()
        graph_tab.addWidget(self.plot)

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
        action_to_open_file.triggered.connect(self.open_file_dialog)
        # Note --->function Raference         ^^^^^^^^^^^^^^^^^^^^^

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

    def open_file_dialog(self):
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
        self.tab_layout.setCurrentIndex(0)

    def activate_tab_1(self):
        self.tab_layout.setCurrentIndex(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
