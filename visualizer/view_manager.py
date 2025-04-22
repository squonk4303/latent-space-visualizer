#!/usr/bin/env python3
import os
import numpy as np

# NOTE: Sucky capitalization on torchvision.models because one is a function and one is a class
from torchvision.models import resnet101, ResNet101_Weights

from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
    QPixmap,
    QWheelEvent,
)

from PyQt6.QtWidgets import (
    QSlider,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QComboBox,
    QWidget,
)

from visualizer import consts, loading, open_dialog, utils
from visualizer.models.segmentation import FCNResNet101
from visualizer.plottables import Plottables, SavableData
from visualizer.plot_widget import PlotWidget
from visualizer.stacked_layout_manager import StackedLayoutManager
import visualizer.models

# Dict for function selection
# Add your desired function with the matched string here
dim_reduction_techs = {
    "TSNE": loading.tsne,
    "PCA": loading.pca,
    "UMAP": loading.umap,
    "TRIMAP": loading.trimap,
    "PACMAP": loading.pacmap,
}


class PrimaryWindow(QMainWindow):
    """
    Primary window of the program.

    This is where all the action happens...
    """

    # =========
    # Callbacks
    # =========
    def wheelEvent(self, ev: QWheelEvent):
        """
        Refers to this event handler:
        https://doc.qt.io/qt-6/qwidget.html#wheelEvent
        """
        # When on the scatter tab
        # (Though it seems matplotlib captures the mouse event)
        if self.tab_layout.currentIndex() == 1:
            # Note that this will trigger slider.valueChanged
            if ev.angleDelta().y() > 0:
                new_value = min(self.slider.maximum(), self.slider.value() + 1)
                self.slider.setValue(new_value)

            elif ev.angleDelta().y() < 0:
                new_value = max(self.slider.minimum(), self.slider.value() - 1)
                self.slider.setValue(new_value)

    # =====
    # Inits
    # =====

    def __init__(self):
        """Constructor for the primary window"""

        # ---------------------------
        # Definitions and Initiations
        # ---------------------------

        super().__init__()

        # Prepare data-holding object
        self.data = SavableData()

        # Set up the tabs in the window
        self.stage_tab = QVBoxLayout()
        graph_tab = QVBoxLayout()

        self.tab_layout = StackedLayoutManager()
        self.tab_layout.add_layout(self.stage_tab)
        self.tab_layout.add_layout(graph_tab)

        # Add buttons to navigate to each tab
        # self.stage_tab_button = QPushButton("Select Data Files")
        # self.graph_tab_button = QPushButton("Visualized Data")
        # self.stage_tab_button.clicked.connect(self.goto_tab(0))
        # self.graph_tab_button.clicked.connect(self.goto_tab(1))

        # tab_buttons_layout = QHBoxLayout()
        # tab_buttons_layout.addWidget(self.stage_tab_button)
        # tab_buttons_layout.addWidget(self.graph_tab_button)

        # And put them in order
        greater_layout = QVBoxLayout()
        # greater_layout.addLayout(tab_buttons_layout)
        greater_layout.addLayout(self.tab_layout)

        # -------
        # Actions
        # -------

        # Quick-saving
        self.quickload_action = QAction("Quickload Plot", parent=self)
        self.quicksave_action = QAction("Quicksave Plot", parent=self)
        self.quickload_action.triggered.connect(self.quickload_wrapper)
        self.quicksave_action.triggered.connect(self.quicksave_wrapper)

        # Save-as...-ing
        self.save_as_action = QAction("Proper Save", parent=self)
        self.load_file_action = QAction("Proper Load", parent=self)
        self.save_as_action.triggered.connect(self.save_to_certain_file_wrapper)
        self.load_file_action.triggered.connect(self.load_file_wrapper)

        # Open the file dialog
        self.action_to_open_file = QAction(consts.OPEN_FILE_LABEL, self)
        self.action_to_open_file.triggered.connect(self.load_model_location)

        # Scroll to next/previous tabs
        self.goto_graph_tab = QAction("&Visualized Data", self)
        self.goto_stage_tab = QAction("&Select Data Files", self)
        # https://doc.qt.io/qt-6/qkeysequence.html#StandardKey-enum
        # PgUp / PgDwn Shortcuts
        self.goto_graph_tab.setShortcut(QKeySequence.StandardKey.MoveToNextPage)
        self.goto_stage_tab.setShortcut(QKeySequence.StandardKey.MoveToPreviousPage)
        # Trigger buttons
        self.goto_graph_tab.triggered.connect(self.goto_tab(1, consts.GRAPH_TITLE))
        self.goto_stage_tab.triggered.connect(self.goto_tab(0, consts.STAGE_TITLE))

        # Initialize Selection Menu in load_tab
        # -----------------------

        self.init_type_selector()
        self.init_reduction_selector()
        self.init_dataset_selection()
        self.init_model_selection()
        self.init_layer_selection()
        self.init_feedback_label()
        self.init_launch_button()

        # ========
        # Plot Tab
        # ========

        # Add the plot widget as a tab
        self.plot = PlotWidget(parent=self)
        self.toolbar = self.plot.make_toolbar()

        # Save/Load Buttons
        quicksave_button = QPushButton("Quicksave Plot")
        quickload_button = QPushButton("Quickload Plot")
        quicksave_button.clicked.connect(self.quicksave_wrapper)
        quickload_button.clicked.connect(self.quickload_wrapper)
        save_as_button = QPushButton("Save As...")
        load_file_button = QPushButton("Load File...")
        save_as_button.clicked.connect(self.save_to_certain_file_wrapper)
        load_file_button.clicked.connect(self.load_file_wrapper)

        # Slider
        self.slider = QSlider(parent=self)
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.set_new_elements_to_display)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setDisabled(True)

        # Organize Widgets for Graph tab
        graph_tab.addWidget(self.plot)
        graph_tab.addWidget(self.toolbar)
        graph_tab.addWidget(self.slider)

        # ---------------------
        # Menu Bar And Submenus
        # ---------------------

        self.init_menu_bar()

        # --------------------
        # Window Configuration
        # --------------------

        self.resize(1080, 640)
        self.setWindowTitle(consts.STAGE_TITLE)
        self.setStatusBar(QStatusBar(self))
        widget = QWidget()
        widget.setLayout(greater_layout)
        self.setCentralWidget(widget)

        # ------
        # Cheats
        # ------

        if consts.flags["dev"]:

            def quick_launch():
                self.data.dataset_location = consts.S_DATASET
                self.data.dim_reduction = "TSNE"
                self.data.layer = consts.LAYER
                self.data.model = FCNResNet101()
                self.data.model_location = consts.MULTILABEL_MODEL
                self.start_cooking_iii()

            quicklaunch_button = QPushButton("Cook")
            quicklaunch_button.clicked.connect(quick_launch)
            self.stage_tab.addWidget(quicklaunch_button)

    # =======
    # Methods
    # =======

    def title_update(self, new_title):
        self.setWindowTitle(new_title)

    def init_model_selection(self):
        self.model_feedback_label = QLabel("<-- Select your trained model's .pth file")
        openfile_button = QPushButton("Select Trained NN Model")
        openfile_button.clicked.connect(self.load_model_location)
        row_model_selection = QHBoxLayout()
        row_model_selection.addWidget(openfile_button)
        row_model_selection.addWidget(self.model_feedback_label)
        self.stage_tab.addLayout(row_model_selection)

    def init_layer_selection(self):
        self.layer_feedback_label = QLabel(
            "<-- Select the layer in your model for the latent space"
        )
        layer_button = QPushButton("Select layer")
        layer_button.clicked.connect(self.find_layer)
        row_layer_selection = QHBoxLayout()
        row_layer_selection.addWidget(layer_button)
        row_layer_selection.addWidget(self.layer_feedback_label)
        self.stage_tab.addLayout(row_layer_selection)

    def init_dataset_selection(self):
        dataset_selection_button = QPushButton("Select Dataset")
        self.dataset_feedback_label = QLabel(
            "<-- Select the folder for the dataset you wish to use"
        )
        dataset_selection_button.clicked.connect(self.find_dataset)
        row_dataset_selection = QHBoxLayout()
        row_dataset_selection.addWidget(dataset_selection_button)
        row_dataset_selection.addWidget(self.dataset_feedback_label)
        self.stage_tab.addLayout(row_dataset_selection)

    def init_launch_button(self):
        self.launch_button = QPushButton("LAUNCH")
        self.launch_button.setDisabled(True)
        self.launch_button.clicked.connect(self.start_cooking_iii)
        self.stage_tab.addWidget(self.launch_button)

    def init_feedback_label(self):
        self.feedback_label = QLabel("")
        self.stage_tab.addWidget(self.feedback_label)

    def init_menu_bar(self):
        """Set up the menu bar, including sub-menus."""

        menubar = self.menuBar()

        # The Greater File Menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.action_to_open_file)
        file_menu.addAction(self.quickload_action)
        file_menu.addAction(self.quicksave_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addAction(self.load_file_action)

        # The Greater Navigation Menu
        navigate_menu = menubar.addMenu("&Tab")
        navigate_menu.addAction(self.goto_graph_tab)
        navigate_menu.addAction(self.goto_stage_tab)

    def init_type_selector(self):

        self.type_select_label = QLabel("<-- Select the desired type of NN model")

        # Dropdown Menu
        type_dropdown = QComboBox(parent=self)
        type_dropdown.addItem("...")
        for model_type in consts.MODEL_TYPES:
            type_dropdown.addItem(model_type)

        # Functionality
        type_dropdown.currentTextChanged.connect(self.suggest_model_type)

        # Layout
        type_select_menu = QHBoxLayout()
        type_select_menu.addWidget(type_dropdown)
        type_select_menu.addWidget(self.type_select_label)

        # Add to stage
        self.stage_tab.addLayout(type_select_menu)

    def suggest_model_type(self, model_type: str):
        """Check modules for model-type, then loads an instance of it into self.model"""
        if hasattr(visualizer.models.segmentation, model_type):
            self.feedback_label.setText(f"You chose model type {model_type}!")
            the_class = getattr(visualizer.models.segmentation, model_type)
            self.data.model = the_class()
            # print(f"Successfully found model {model_type}, {the_class}")
            self.try_to_activate_launch_button()

        # elif hasattr(models.whatever, model_type):
        #     self.feedback_label.setText("You sure chose " + model_type)
        #     self.model = getattr(models.whatever, model_type)()
        #     self.try_to_load_model()

        elif model_type == "...":
            print(f"Model is {model_type}? Whatever. I don't care.")

        else:
            raise ValueError(
                f"Woah. Model type {model_type} wasn't supposed to be selectable."
            )

    def init_reduction_selector(self):
        self.reduction_select_label = QLabel(
            "<-- Select the desired reduction technique"
        )

        # Dropdown Menu
        reduction_dropdown = QComboBox(parent=self)
        reduction_dropdown.addItem("...")
        # for technique in dim_reduction_techs.keys():
        #     reduction_dropdown.addItem(technique)
        reduction_dropdown.addItem("t-SNE")
        reduction_dropdown.addItem("P.C.A.")
        reduction_dropdown.addItem("~UMAP")
        reduction_dropdown.addItem("TRI-MAP")
        reduction_dropdown.addItem("PA¢CMAP")
        reduction_dropdown.addItem("SE.GMEN.TAT.ION")
        reduction_dropdown.addItem("CLASS!IFICATION_24")
        reduction_dropdown.addItem("bogus")

        # Functionality
        reduction_dropdown.currentTextChanged.connect(self.suggest_dim_reduction)

        # Layout
        reduction_select_menu = QHBoxLayout()
        reduction_select_menu.addWidget(reduction_dropdown)
        reduction_select_menu.addWidget(self.reduction_select_label)

        # Add to stage
        self.stage_tab.addLayout(reduction_select_menu)

    def suggest_dim_reduction(self, text: str):
        # Reformatting text without special characters like - _ *
        standardized_input = ("".join(filter(str.isalpha, text))).upper()

        # Checking if the chosen function exists in list of functions and then call it
        if standardized_input in dim_reduction_techs:
            # Update self.data.technique to be the matching function in the dict
            self.data.dim_reduction = standardized_input
            self.feedback_label.setText(
                f"You chose dim reduction technique {standardized_input}"
            )
            self.try_to_activate_launch_button()
        elif standardized_input != "":
            raise RuntimeError(
                f"Selected technique {standardized_input} not found in {dim_reduction_techs}"
            )

    def set_new_elements_to_display(self, value):
        """
        Change the tuple(picture, coordinate, mask, &c) selected in the scatter tab.

        Called whenever the slider changes, by user or other means.
        """
        self.plot.new_tuple(
            value,
            self.data.labels,
            self.data.paths,
            self.data.two_dee,
            self.data.masks,
            self.data.model.colormap,
        )

    def load_model_location(self):
        """
        Open dialog for finding a trained neural-net-model, and inform user if successful.

        For use in buttons and actions.
        """
        model_path = open_dialog.for_trained_model_file(parent=self)

        if model_path:
            self.data.model_location = model_path
            self.model_feedback_label.setText("You chose: " + str(model_path))
            self.feedback_label.setText("You chose: " + str(model_path))
            self.try_to_activate_launch_button()

    def try_to_activate_launch_button(self):
        """
        Evaluate and enact whether the go-for-it-button should be enabled.

        Meaning if dataset, layer, model, and model type are selected, the button is activated,
        and if any of these are found to be insufficient, the button is deactivated.
        """
        # @Wilhelmsen: Check dataset by there being more than 3 images in there instead
        # 3 because that's how many the t-sne needs (or 2, I'm not sure)
        dataset_alright = (
            os.path.isdir(self.data.dataset_location)
            if self.data.dataset_location is not None
            else False
        )
        dim_reduction_alright = bool(self.data.dim_reduction)
        layer_alright = bool(self.data.layer)
        model_alright = hasattr(self.data.model, "state_dict")
        model_location_alright = bool(self.data.model_location)

        # print(
        #     f"dataset location: {self.data.dataset_location} bool: {dataset_alright}\n"
        #     f"dim reduction:    {self.data.dim_reduction}  bool: {dim_reduction_alright}\n"
        #     f"layer:            {self.data.layer}  bool: {layer_alright}\n"
        #     f"model location:   {self.data.model_location} bool: {model_location_alright}\n"
        #     f"model:                    bool: {model_alright}\n"
        # )

        should_be_enabled = bool(
            dataset_alright
            and dim_reduction_alright
            and layer_alright
            and model_alright
            and model_location_alright
        )

        self.launch_button.setDisabled(not should_be_enabled)

    def find_dataset(self):
        """
        Open dialog for finding dataset, and inform user if successful.

        Is intended to be used to be used for directories containing the datasets.
        For use in buttons and actions.
        """
        self.data.dataset_location = open_dialog.for_directory(
            parent=self, caption=consts.DATASET_DIALOG_CAPTION
        )
        if self.data.dataset_location:
            paths = utils.grab_image_paths_in_dir(self.data.dataset_location)
            text = str(self.data.dataset_location) + ", length: " + str(len(paths))
            self.dataset_feedback_label.setText(text)
            self.feedback_label.setText(text)
            self.try_to_activate_launch_button()

    def find_layer(self):
        """
        @Linnea: Complete this function and add a small docstring
                 Pretty please
        """
        print("Hardcoding a model in; yet TODO; hardcode it out again")
        self.data.model = FCNResNet101()
        self.data.model.load(consts.MULTILABEL_MODEL)
        selected_layer = open_dialog.for_layer_select(
            self.data.model, "SELECT LAYER", parent=self
        )
        print("*** selected_layer:", selected_layer)
        if selected_layer:
            self.data.layer = selected_layer
            self.layer_feedback_label.setText("You chose " + selected_layer)
            self.feedback_label.setText("You chose " + selected_layer)
            self.try_to_activate_launch_button()

    def start_cooking_iii(self):
        # Try to load the trained model
        # On failure, give the user some constructive feedback
        # @Wilhelmsen: Better error handling please!
        # Try to prevent the program from crashing on bad file
        # Maybe just by ensuring .pth as file extension...
        try:
            self.data.model.load(self.data.model_location)
        except RuntimeError as e:
            print(
                f"Tried and failed to load model! "
                f"type: {type(self.data.model)}, "
                f"location: {self.data.model.location}, "
                f"error message: {e}"
            )
            # @Wilhelmsen: Give the user a little *better* constructive feedback
            self.feedback_label.setText(e)

        # @Wilhelmsen: This could be an iglob
        self.data.paths = utils.grab_image_paths_in_dir(self.data.dataset_location)
        reduced_data, paths, labels, masks = loading.preliminary_dim_reduction_iii(
            self.data.model, self.data.layer, self.data.paths
        )
        self.try_to_load_model()
        # @Wilhelmsen: Move this assertion to tests
        assert len(reduced_data) == len(paths) == len(labels) == len(masks)

        # Normalize array
        # @Wilhelmsen: This normalizes for the whole matrix at once,
        #              As opposed to for each axis, which is what I want
        #              And it also doesn't at all work
        arr = dim_reduction_techs[self.data.dim_reduction](reduced_data)
        plottable_data = arr / np.min(arr) / (np.max(arr) / np.min(arr))

        self.data.labels = labels
        self.data.masks = masks
        self.data.paths = paths
        self.data.two_dee = plottable_data
        self.utilize_data()

    def utilize_data(self):
        # @Wilhelmsen: Refer to these by keywords
        self.plot.the_plottables(
            self.data.labels,
            self.data.paths,
            self.data.two_dee,
            self.data.masks,
            self.data.model.colormap,
        )
        # Set slider limits
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.data.paths) - 1)
        self.slider.setDisabled(False)

    def goto_tab(self, n, titleupdate="Missing Title"):
        """
        Return a function which changes to tab specified by argument.

        The returned function is a callback function used as a parameter in
        f.ex. buttons and actions.
        """

        def func():
            self.title_update(titleupdate)
            self.tab_layout.setCurrentIndex(n)

        return func

    def quickload_wrapper(self):
        """Load from default save file directly to .data"""
        self.data = loading.quickload()

        if self.data.model is not None:
            self.utilize_data()
        else:
            # @Wilhelmsen: Find something to do with this
            print("There's nothing here! TODO")

    def quicksave_wrapper(self):
        """Save .data directly to default save file."""
        loading.quicksave(self.data)

    def save_to_certain_file_wrapper(self):
        """Open dialog for a file for which to save .data"""
        save_location = loading.save_to_user_selected_file(self.data, parent=self)
        if save_location:
            self.feedback_label.setText("Saved to " + save_location)

    def load_file_wrapper(self):
        """Open dialog for file from which to load into .data"""
        self.data = loading.load_by_dialog(parent=self)

        if self.data is not None:
            self.utilize_data()
