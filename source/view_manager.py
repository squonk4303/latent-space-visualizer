#!/usr/bin/env python3
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
)
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import consts
import loading
from stacked_layout_manager import StackedLayoutManager
from plot_widget import PlotWidget


class PrimaryWindow(QMainWindow):
    """
    Primary window of the program
    This is where all the action happens
    """
    def __init__(self, *args, **kwargs):
        """ Constructor for the primary window """
        super().__init__(*args, **kwargs)

        # --- Layout ---
        # --- Definitions
        greater_layout = QVBoxLayout()
        self.tab_layout = StackedLayoutManager()

        tab_buttons_layout = QHBoxLayout()
        empty_tab = QVBoxLayout()
        graph_tab = QVBoxLayout()

        self.openfile_button = QPushButton(consts.OPEN_FILE_LABEL)
        self.TEMP_button = QPushButton("-->")

        self.empty_tab_button = QPushButton("0")
        self.graph_tab_button = QPushButton("1")

        # --- Initiations
        self.initiate_menu_bar()

        # --- Signals
        self.TEMP_button.clicked.connect(self.tab_layout.scroll_forth)
        self.openfile_button.clicked.connect(self.load_model_file)
        self.empty_tab_button.clicked.connect(self.activate_tab_0)
        self.graph_tab_button.clicked.connect(self.activate_tab_1)

        # --- Layout Organization
        greater_layout.addLayout(tab_buttons_layout)
        greater_layout.addLayout(self.tab_layout)

        tab_buttons_layout.addWidget(self.empty_tab_button)
        tab_buttons_layout.addWidget(self.graph_tab_button)

        empty_tab.addWidget(self.openfile_button)
        empty_tab.addWidget(self.TEMP_button)

        # Adds the plot widget as a tab
        self.plot = PlotWidget()
        toolbar = self.plot.make_toolbar()

        graph_tab.addWidget(self.plot)
        graph_tab.addWidget(toolbar)

        self.tab_layout.add_layout(empty_tab)
        self.tab_layout.add_layout(graph_tab)

        # --- Window Configuration ---
        self.resize(650, 450)
        self.setWindowTitle(consts.WINDOW_TITLE)
        widget = QWidget()
        widget.setLayout(greater_layout)
        self.setCentralWidget(widget)

    def initiate_menu_bar(self):
        """Set up the top menu-bar, its sub-menues, actions, and signals."""
        menubar = self.menuBar()

        # Action which opens the file dialog
        action_to_open_file = QAction(consts.OPEN_FILE_LABEL, self)
        action_to_open_file.setStatusTip(consts.STATUS_TIP_TEMP)
        action_to_open_file.setShortcut(QKeySequence("Ctrl+O"))
        action_to_open_file.triggered.connect(self.load_model_file)

        # Actions to scroll to next/previous tabs
        next_tab = QAction("TEMP: &Next tab")
        next_tab.setStatusTip(consts.STATUS_TIP_TEMP)
        prev_tab = QAction("TEMP: &Previous tab")
        prev_tab.setStatusTip(consts.STATUS_TIP_TEMP)

        next_tab.triggered.connect(self.tab_layout.scroll_forth)
        prev_tab.triggered.connect(self.tab_layout.scroll_back)

        # The Greater File Menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(action_to_open_file)

        # The Greater Navigaiton Menu
        self.navigate_menu = menubar.addMenu("&Tab")
        self.navigate_menu.addAction(next_tab)
        self.navigate_menu.addAction(prev_tab)

        # Make these variables available to class namespace
        self.action_to_open_file = action_to_open_file
        self.next_tab = next_tab
        self.prev_tab = prev_tab
        self.file_menu = file_menu

    def load_model_file(self):
        handler = loading.FileDialogManager(self)
        model_path = handler.find_trained_model_file()
        categories = ["skin"]
        # If user cancels dialog, does nothing
        if model_path:
            # loaded_model = loading.get_model(model_path, categories)
            # loading.layer_summary(loaded_model, 1, 2)
            # reduced_data = loading.reduce_data(model_path, categories)
            reduced_data = loading.the_whole_enchilada()
            print(reduced_data)
            self.plot.plot_from_2d(reduced_data)

    def activate_tab_0(self):
        self.tab_layout.setCurrentIndex(0)

    def activate_tab_1(self):
        self.tab_layout.setCurrentIndex(1)
