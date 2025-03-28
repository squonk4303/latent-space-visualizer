#!/usr/bin/env python3
import torch

from PyQt6.QtGui import (
    QAction,
    QKeySequence,
    QPixmap,
)

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from visualizer import consts, loading, open_dialog, utils
from visualizer.plottables import Plottables
from visualizer.external.fcn import FCNResNet101
from visualizer.plot_widget import PlotWidget
from visualizer.stacked_layout_manager import StackedLayoutManager
from visualizer.consts import DR_technique as Technique
from visualizer.loading import apply_tsne as t_sne

class PrimaryWindow(QMainWindow):
    """
    Primary window of the program
    This is where all the action happens
    """

    def __init__(self, *args, **kwargs):
        """Constructor for the primary window"""
        super().__init__(*args, **kwargs)

        # Prepare data-holding object
        self.data = Plottables()

        # Set up the tabs in the window
        start_tab = QVBoxLayout()
        graph_tab = QVBoxLayout()

        self.tab_layout = StackedLayoutManager()
        self.tab_layout.add_layout(start_tab)
        self.tab_layout.add_layout(graph_tab)

        # Add buttons to navigate to each tab
        self.start_tab_button = QPushButton("0")
        self.graph_tab_button = QPushButton("1")
        self.start_tab_button.clicked.connect(self.callable_goto_tab(0))
        self.graph_tab_button.clicked.connect(self.callable_goto_tab(1))

        tab_buttons_layout = QHBoxLayout()
        tab_buttons_layout.addWidget(self.start_tab_button)
        tab_buttons_layout.addWidget(self.graph_tab_button)

        # And put them in order
        greater_layout = QVBoxLayout()
        greater_layout.addLayout(tab_buttons_layout)
        greater_layout.addLayout(self.tab_layout)

        # --- Declare Actions ---

        # Quick-saving
        # quickappend_action = QAction("Quickappend Plot", parent=self)
        self.quickload_action = QAction("Quickload Plot", parent=self)
        self.quicksave_action = QAction("Quicksave Plot", parent=self)

        self.quickload_action.triggered.connect(self.quickload_wrapper)
        self.quicksave_action.triggered.connect(self.quicksave_wrapper)

        # Saving-as
        self.save_as_action = QAction("Proper Save", parent=self)
        self.load_file_action = QAction("Proper Load", parent=self)

        self.save_as_action.triggered.connect(self.save_to_certain_file_wrapper)
        self.load_file_action.triggered.connect(self.load_file_wrapper)

        # Seu up the menu bar and submenus
        self.initiate_menu_bar()

        # --- Initialize start screen ---

        # Row For Model Selection
        # -----------------------
        self.model_feedback_label = QLabel("<-- File dialog for .pth")
        openfile_button = QPushButton("Select Trained NN Model")
        openfile_button.clicked.connect(self.load_model_file)

        row_model_selection = QHBoxLayout()
        row_model_selection.addWidget(openfile_button)
        row_model_selection.addWidget(self.model_feedback_label)

        # Row For Category Selection
        # --------------------------
        self.category_feedback_label = QLabel(
            "<-- Either a text input or a menu where you get to choose between a lot of text-options-- "
            "and get to choose multiple. Consider QListView or QListWidget for this."
        )
        category_button = QPushButton("Select Categories")

        row_category_selection = QHBoxLayout()
        row_category_selection.addWidget(category_button)
        row_category_selection.addWidget(self.category_feedback_label)

        # Row For Layer Selection
        # -----------------------
        self.layer_feedback_label = QLabel("<-- You know-- something to select layers")
        layer_button = QPushButton("Select layer")

        row_layer_selection = QHBoxLayout()
        row_layer_selection.addWidget(layer_button)
        row_layer_selection.addWidget(self.layer_feedback_label)

        # Row For Dataset Selection
        # -------------------------
        dataset_selection_button = QPushButton("Select Dataset")
        self.dataset_feedback_label = QLabel(
            "<-- Just a file dialog for directories should be fine"
        )
        dataset_selection_button.clicked.connect(self.find_dataset)

        row_dataset_selection = QHBoxLayout()
        row_dataset_selection.addWidget(dataset_selection_button)
        row_dataset_selection.addWidget(self.dataset_feedback_label)

        # Row For Single Picture Selection
        # --------------------------------
        # Note on pixmap from https://doc.qt.io/qt-6/qpixmap.html#details
        # QPixmap is designed and optimized for showing images on screen
        self.single_image_label = QLabel("<-- Normal file dialog for images")
        self.single_image_thumb_label = QLabel()
        self.single_image_thumb_label.setPixmap(QPixmap("assets/default_pic.png"))
        single_image_button = QPushButton("Select Image")
        single_image_button.clicked.connect(self.find_picture)

        # Has a row on top:     [button]  [label]
        # Then one underneath:  [image thumbnail]
        single_image_subrow = QHBoxLayout()
        single_image_subrow.addWidget(single_image_button)
        single_image_subrow.addWidget(self.single_image_label)

        row_single_image = QVBoxLayout()
        row_single_image.addLayout(single_image_subrow)
        row_single_image.addWidget(self.single_image_thumb_label)

        # Button Which Confirms Input and Goes to Graph Tab
        # -------------------------------------------------
        self.register_stuff_button = QPushButton("Go for it~!")
        self.register_stuff_button.setDisabled(True)
        self.register_stuff_button.clicked.connect(self.start_cooking)

        # Put them all in order
        start_tab.addLayout(row_model_selection)
        start_tab.addLayout(row_category_selection)
        start_tab.addLayout(row_dataset_selection)
        start_tab.addLayout(row_layer_selection)
        start_tab.addLayout(row_single_image)
        start_tab.addWidget(self.register_stuff_button)

        # --- Plot Tab ---

        # Add the plot widget as a tab
        self.plot = PlotWidget(parent=self)
        self.toolbar = self.plot.make_toolbar()

        quicksave_button = QPushButton("Quicksave Plot")
        quickload_button = QPushButton("Quickload Plot")
        quicksave_button.clicked.connect(self.quicksave_wrapper)
        quickload_button.clicked.connect(self.quickload_wrapper)

        save_as_button = QPushButton("Save As...")
        load_file_button = QPushButton("Load File...")
        save_as_button.clicked.connect(self.save_to_certain_file_wrapper)
        load_file_button.clicked.connect(self.load_file_wrapper)

        graph_tab.addWidget(self.plot)
        graph_tab.addWidget(self.toolbar)
        graph_tab.addWidget(quickload_button)
        graph_tab.addWidget(quicksave_button)
        graph_tab.addWidget(save_as_button)
        graph_tab.addWidget(load_file_button)

        # --- Window Configuration ---

        self.resize(650, 450)
        self.setWindowTitle(consts.WINDOW_TITLE)
        self.setStatusBar(QStatusBar(self))
        widget = QWidget()
        widget.setLayout(greater_layout)
        self.setCentralWidget(widget)

        # --- Cheats ---

        dev_button = QPushButton("Cheat")
        dev_button.clicked.connect(self.start_cooking)
        start_tab.addWidget(dev_button)

        if consts.flags["dev"]:
            self.start_cooking()  # <-- Just goes ahead and starts cooking

    def initiate_menu_bar(self):
        """Set up the top menu-bar, its sub-menus, actions, and signals."""
        menubar = self.menuBar()

        # Action which opens the file dialog
        action_to_open_file = QAction(consts.OPEN_FILE_LABEL, self)
        action_to_open_file.triggered.connect(self.load_model_file)

        # Actions to scroll to next/previous tabs
        next_tab = QAction("&Next tab", self)
        prev_tab = QAction("&Previous tab", self)
        # https://doc.qt.io/qt-6/qkeysequence.html#StandardKey-enum
        next_tab.setShortcut(QKeySequence.StandardKey.MoveToNextPage)
        prev_tab.setShortcut(QKeySequence.StandardKey.MoveToPreviousPage)

        next_tab.triggered.connect(self.tab_layout.scroll_forth)
        prev_tab.triggered.connect(self.tab_layout.scroll_back)

        # The Greater File Menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(action_to_open_file)

        file_menu.addAction(self.quickload_action)
        file_menu.addAction(self.quicksave_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addAction(self.load_file_action)

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
        model_path = open_dialog.for_trained_model_file(parent=self)
        if model_path:
            self.model_feedback_label.setText("You chose: " + model_path)

    def find_dataset(self):
        """
        Open dialog for finding dataset.

        Called by dataset_selection_button.
        """
        dataset_dir = open_dialog.for_directory(parent=self)
        if dataset_dir:
            self.dataset_feedback_label.setText("You found: " + dataset_dir)

    def find_picture(self):
        """Get image from user choice and prepare it for processing."""
        image_path = open_dialog.for_image_file(parent=self)
        if image_path:
            # @Wilhelmsen: Yet to check validity and resize image
            self.single_image_label.setText(image_path)
            self.single_image_thumb_label.setPixmap(QPixmap(image_path))

            # Start the process of dim.reducing the image
            # Note that we wrap image_path in a tuple
            _ = loading.dataset_to_tensors((image_path,))
            # And take it through t-SNE just for good measure too
            # Or not...
            # Maybe it's best to leave that for whwn things are plotted onto the graph...
            # I'm wondering if t-SNED coords shouldn't be stored over there anyways.
            # Then again it could be helpful to have them put over there immediately...

    def technique_loader(features, target_dimensionality=2, reduction=Technique.T_SNE):
        
         #Added a switch for later implementation of more reduction methods
        match reduction:
            case Technique.T_SNE: #Maybe have this be based off of enums  instead?
                return t_sne(features,target_dimensionality)
            case Technique.PCA:
                return None # TBI (TO BE IMPLEMENTED)
            case Technique.UMAP:    
                return None # TBI
            case Technique.TRIMAP:
                return None # TBI
            case Technique.PACMAP:
                return None # TBI
            case _:                 # Default case
                return None
                raise RuntimeError("No reduction technique selected!")

    # @Wilhelmsen: This should be MOCKED and harangued
    def start_cooking(self):
        # Make sure to define device
        consts.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare data
        categories = consts.DEFAULT_MODEL_CATEGORIES
        self.data.selected_layer = "layer4"
        self.data.model = FCNResNet101(categories)
        self.data.model.load(consts.TRAINED_MODEL)
        dataset_paths = utils.grab_image_paths_in_dir(consts.MEDIUM_DATASET)
        image_tensors = loading.dataset_to_tensors(dataset_paths)
        _ = loading.dataset_to_tensors((consts.GRAPHICAL_IMAGE,))

        print("".join([f"tensor: {t.shape}\n" for t in image_tensors]))

        self.data.dataset_intermediary = loading.preliminary_dim_reduction(
            self.data.model, image_tensors, self.data.selected_layer
        )

        print(
            "".join(
                [
                    f"data.dataset_intermediary: {t.shape}\n"
                    for t in self.data.dataset_intermediary
                ]
            )
        )

        self.data.dataset_plottable = loading.apply_tsne(self.data.dataset_intermediary)
        # self.dataset_plottable = self.technique_loader(reduced_features)
        # tsned_single = loading.apply_tsne(single_image_tensor)

        print(
            "".join(
                [
                    f"self.data.dataset_plottable: {t}\n"
                    for t in self.data.dataset_plottable
                ]
            )
        )

        self.plot.plot_from_2d(self.data.dataset_plottable)
        # self.plot.plot_from_2d(tsned_single)

        self.quicksave_wrapper()

    def callable_goto_tab(self, n):
        """Return a function which changes to tab specified by argument."""

        def f():
            self.tab_layout.setCurrentIndex(n)

        return f

    def quickload_wrapper(self):
        self.data = loading.quickload()

        if self.data.dataset_plottable is not None:
            self.plot.plot_from_2d(self.data.dataset_plottable)
        else:
            print("There's nothing here! TODO")

    def quicksave_wrapper(self):
        loading.quicksave(self.data)

    def save_to_certain_file_wrapper(self):
        _ = loading.save_to_user_selected_file(self.data, parent=self)


    def load_file_wrapper(self):
        self.data = loading.load_by_dialog(parent=self)

        if self.data is not None:
            self.plot.plot_from_2d(self.data.dataset_plottable)
