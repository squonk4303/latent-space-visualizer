#!/usr/bin/env python3
import os

# NOTE: Sucky capitalization on torchvision.models because one is a function and one is a class
from torchvision.models import resnet101, ResNet101_Weights

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
from visualizer.consts import DR_technique as Technique
from visualizer.loading import apply_tsne as t_sne
from visualizer.models.segmentation import FCNResNet101
from visualizer.plottables import Plottables, SavableData
from visualizer.plot_widget import PlotWidget
from visualizer.stacked_layout_manager import StackedLayoutManager


class PrimaryWindow(QMainWindow):
    """
    Primary window of the program.

    This is where all the action happens...
    """

    def __init__(self):
        """Constructor for the primary window"""

        # ---------------------------
        # Definitions and Initiations
        # ---------------------------

        super().__init__()

        # Prepare data-holding object
        self.data = SavableData()

        # Set up the tabs in the window
        self.start_tab = QVBoxLayout()
        graph_tab = QVBoxLayout()

        self.tab_layout = StackedLayoutManager()
        self.tab_layout.add_layout(self.start_tab)
        self.tab_layout.add_layout(graph_tab)

        # Add buttons to navigate to each tab
        self.start_tab_button = QPushButton("0")
        self.graph_tab_button = QPushButton("1")
        self.start_tab_button.clicked.connect(self.goto_tab(0))
        self.graph_tab_button.clicked.connect(self.goto_tab(1))

        tab_buttons_layout = QHBoxLayout()
        tab_buttons_layout.addWidget(self.start_tab_button)
        tab_buttons_layout.addWidget(self.graph_tab_button)

        # And put them in order
        greater_layout = QVBoxLayout()
        greater_layout.addLayout(tab_buttons_layout)
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
        self.action_to_open_file.triggered.connect(self.load_model_file)

        # Scroll to next/previous tabs
        self.next_tab = QAction("&Next tab", self)
        self.prev_tab = QAction("&Previous tab", self)
        # https://doc.qt.io/qt-6/qkeysequence.html#StandardKey-enum
        self.next_tab.setShortcut(QKeySequence.StandardKey.MoveToNextPage)
        self.prev_tab.setShortcut(QKeySequence.StandardKey.MoveToPreviousPage)
        self.next_tab.triggered.connect(self.tab_layout.scroll_forth)
        self.prev_tab.triggered.connect(self.tab_layout.scroll_back)

        # Initialize start screen
        # -----------------------

        self.init_model_selection()
        self.init_layer_selection()
        self.init_dataset_selection()
        self.init_feedback_label()
        self.init_go_for_it_button()

        # ========
        # Plot Tab
        # ========

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

        # ---------------------
        # Menu Bar And Submenus
        # ---------------------

        self.init_menu_bar()

        # --------------------
        # Window Configuration
        # --------------------

        self.resize(1080, 640)
        self.setWindowTitle(consts.WINDOW_TITLE)
        self.setStatusBar(QStatusBar(self))
        widget = QWidget()
        widget.setLayout(greater_layout)
        self.setCentralWidget(widget)

        # Cheats
        # ------

        if consts.flags["dev"]:
            quicklaunch_button = QPushButton("Cook I")
            quicklaunch_button.clicked.connect(self.quick_launch)
            self.start_tab.addWidget(quicklaunch_button)

    def quick_launch(self):
        self.data.model = os.path.join(
            consts.REPO_DIR, "models.ignore/rgb-aug0/best_model.pth"
        )
        self.data.model = FCNResNet101()
        self.data.model.load(consts.MULTILABEL_MODEL)
        self.data.layer = "layer4"
        self.data.dataset_location = os.path.join(
            consts.REPO_DIR, "pics/dataset_w_json"
        )
        self.start_cooking_iii()

    def init_model_selection(self):
        self.model_feedback_label = QLabel("<-- File dialog for .pth")
        openfile_button = QPushButton("Select Trained NN Model")
        openfile_button.clicked.connect(self.load_model_file)
        row_model_selection = QHBoxLayout()
        row_model_selection.addWidget(openfile_button)
        row_model_selection.addWidget(self.model_feedback_label)
        self.start_tab.addLayout(row_model_selection)

    def init_layer_selection(self):
        self.layer_feedback_label = QLabel("<-- You know-- something to select layers")
        layer_button = QPushButton("Select layer")
        layer_button.clicked.connect(self.find_layer)
        row_layer_selection = QHBoxLayout()
        row_layer_selection.addWidget(layer_button)
        row_layer_selection.addWidget(self.layer_feedback_label)
        self.start_tab.addLayout(row_layer_selection)

    def init_dataset_selection(self):
        dataset_selection_button = QPushButton("Select Dataset")
        self.dataset_feedback_label = QLabel("<-- Just a file dialog for directories")
        dataset_selection_button.clicked.connect(self.find_dataset)
        row_dataset_selection = QHBoxLayout()
        row_dataset_selection.addWidget(dataset_selection_button)
        row_dataset_selection.addWidget(self.dataset_feedback_label)
        self.start_tab.addLayout(row_dataset_selection)

    def init_go_for_it_button(self):
        self.go_for_it_button = QPushButton("Go for it~!")
        self.go_for_it_button.setDisabled(True)
        self.go_for_it_button.clicked.connect(self.start_cooking_iii)
        self.start_tab.addWidget(self.go_for_it_button)

    def init_feedback_label(self):
        self.feedback_label = QLabel("")
        self.start_tab.addWidget(self.feedback_label)

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
        navigate_menu.addAction(self.next_tab)
        navigate_menu.addAction(self.prev_tab)

    def load_model_file(self):
        """
        Open dialog for finding a trained neural-net-model, and inform user if successful.

        For use in buttons and actions.
        """
        model_path = open_dialog.for_trained_model_file(parent=self)
        if model_path:
            # @Wilhelmsen: Better error handling please!
            # Try to prevent the program from crashing on bad file
            # Maybe just by ensuring .pth as file extension...
            try:
                self.data.model = FCNResNet101()
                self.data.model.load(model_path)
                self.model_feedback_label.setText("You chose: " + model_path)
                self.feedback_label.setText("You chose: " + model_path)
            except RuntimeError as e:
                print(f"something went wrong; {e}")
                self.feedback_label.setText(e)

            self.try_to_activate_goforit_button()

    def try_to_activate_goforit_button(self):
        """
        Evaluate and enact whether the go-for-it-button should be enabled.

        Meaning if dataset, layer and model are selected, the button is activated,
        and if any of these are found to be insufficient, the button is deactivated.
        """
        model_alright = hasattr(self.data.model, "state_dict")
        dataset_alright = os.path.isdir(self.data.dataset_location)
        categories_alright = (
            len(self.data.model.categories) > 0
            if hasattr(self.data.model, "categories")
            else False
        )

        should_be_disabled = bool(
            not (
                model_alright
                and categories_alright
                and self.data.layer
                and dataset_alright
            )
        )
        # self.go_for_it_button.setDisabled(should_be_disabled)
        self.go_for_it_button.setDisabled(should_be_disabled)

    def find_dataset(self):
        """
        Open dialog for finding dataset, and inform user if successful.

        Is intended to be used to be used for directories containing the datasets.
        For use in buttons and actions.
        """
        self.data.dataset_location = open_dialog.for_directory(parent=self)
        if self.data.dataset_location:
            paths = utils.grab_image_paths_in_dir(self.data.dataset_location)
            text = self.data.dataset_location + ", length: " + str(len(paths))
            self.dataset_feedback_label.setText(text)
            self.feedback_label.setText(text)
            self.try_to_activate_goforit_button()

    def find_picture(self):
        """
        Open dialog for finding picture, and inform user if successful.

        For use in buttons and actions. Probably deprecated.
        """
        image_path = open_dialog.for_image_file(parent=self)
        if image_path:
            # @Wilhelmsen: Yet to check image validity
            # @Wilhelmsen: Yet to resize image for better display in GUI
            self.single_image_label.setText(image_path)
            self.single_image_thumb_label.setPixmap(QPixmap(image_path))

    def find_layer(self):
        """
        @Linnea: Complete this function and add a small docstring
                 Pretty please
        """
        selected_layer = "layer4"
        if selected_layer:
            self.data.layer = selected_layer
            self.layer_feedback_label = "You chose " + selected_layer
            self.try_to_activate_goforit_button()

    def technique_loader(features, target_dimensionality=2, reduction=Technique.T_SNE):
        """
        @Linnea: Write a docstring for this method.
        """

        # Added a switch for later implementation of more reduction methods
        match reduction:
            case Technique.T_SNE:
                # @Linnea: Better to rename the function in its definition and also refer to it by namespace
                return t_sne(features, target_dimensionality)
            case Technique.PCA:
                return None  # TBI (TO BE IMPLEMENTED)
            case Technique.UMAP:
                return None  # TBI
            case Technique.TRIMAP:
                return None  # TBI
            case Technique.PACMAP:
                return None  # TBI
            case _:  # Default case
                return None
                raise RuntimeError("No reduction technique selected!")

    def start_cooking_iii(self):
        # @Wilhelmsen: This could be an iglob
        image_locations = utils.grab_image_paths_in_dir(self.data.dataset_location)
        reduced_data, paths, labels, masks = loading.preliminary_dim_reduction_iii(
            self.data.model, self.data.layer, image_locations
        )
        # @Wilhelmsen: Move this assertion to tests
        assert len(reduced_data) == len(paths) == len(labels) == len(masks)

        plottable_data = loading.apply_tsne(reduced_data)
        # @Wilhelmsen: This is where a data normalization would take place
        self.data.labels = labels
        self.data.masks = masks
        self.data.paths = paths
        self.data.two_dee = plottable_data
        self.plot.the_plottables(
            self.data.labels, self.data.paths, self.data.two_dee, self.data.masks
        )

    def start_cooking_brains(self):
        """
        Walk through the dim-reduction process with the brain dataset.

        For use in testing/development. Deletion pending.
        """
        # -----------------
        # Loading the Model
        # -----------------

        # @Wilhelmsen: Easiest way to import a pretrained model but that's not what's up
        weights = ResNet101_Weights.DEFAULT
        self.data.model = resnet101(weights=weights)
        self.data.model.eval()

        # ------------------------------------
        # Ordering image paths with categories
        # ------------------------------------

        # Prepare data
        dirs = [
            os.path.join(consts.REPO_DIR, "pics/brains/category0/PNG/"),
            os.path.join(consts.REPO_DIR, "pics/brains/category1/PNG/"),
            os.path.join(consts.REPO_DIR, "pics/brains/category2/PNG/"),
            os.path.join(consts.REPO_DIR, "pics/brains/category3/PNG/"),
        ]
        self.data.layer = "layer4"

        # NOTE that categories and dirs have to be lined up to correspond in their discrete lists
        categories = ["category0", "category1", "category2", "category3"]
        pics_by_category = {
            category: utils.grab_image_paths_in_dir(pics)
            for category, pics in zip(categories, dirs)
        }

        # ----------------
        # Extract Features
        # ----------------
        for label, files in pics_by_category.items():
            # @Wilhelmsen: Change the interface for old_plottables in the model. "self.data.whatever" is sucks
            paths, features = loading.preliminary_dim_reduction_2(
                self.data.model, self.data.layer, label, files
            )

            # Such as this, where the lists are numpy.ndarrays:
            # self.data.old_plottables["category0"] = [
            #     Plottables("/file/path_0", [1, 2, 3, 4]),
            #     Plottables("/file/path_5", [6, 7, 8, 9]),
            #     ... ,
            # ]
            self.data.old_plottables[label] = [
                Plottables(p, f) for p, f in zip(paths, features)
            ]

        # ---------------------------------------------------------------------------
        # t-SNE & Plot
        # ---------------------------------------------------------------------------
        self.plot.with_tsne(self.data.old_plottables)

    def start_cooking(self):
        """
        Walk through the dim-reduction process with pre-determined parameters.

        Mostly for use in testing/development.
        """
        self.data.layer = "layer4"
        self.data.model = FCNResNet101()

        self.data.model.load("models.ignore/rgb-aug0/best_model.pth")
        # self.data.model.load("models.ignore/rgb-aug0/checkpoint.pth")
        # self.data.model.load("models.ignore/rgb-aug1/best_model.pth")
        # self.data.model.load("models.ignore/rgb-aug1/checkpoint.pth")
        # self.data.model.load("models.ignore/rgb-aug2/best_model.pth")
        # self.data.model.load("models.ignore/rgb-aug2/checkpoint.pth")

        print("--- self.data.model:", self.data.model)
        dataset = os.path.join(consts.REPO_DIR, "pics/dataset_w_json")
        dataset_paths = utils.grab_image_paths_in_dir(dataset)
        image_tensors = loading.dataset_to_tensors(dataset_paths)

        self.data.dataset_intermediary = loading.preliminary_dim_reduction(
            self.data.model, image_tensors, self.data.layer
        )
        self.data.dataset_plottable = loading.apply_tsne(self.data.dataset_intermediary)
        self.plot.plot_from_2d(self.data.dataset_plottable)
        self.quicksave_wrapper()

    def goto_tab(self, n):
        """
        Return a function which changes to tab specified by argument.

        The returned function is a callback function used as a parameter in
        f.ex. buttons and actions.
        """
        return lambda: self.tab_layout.setCurrentIndex(n)

    def quickload_wrapper(self):
        """Load from default save file directly to .data"""
        self.data = loading.quickload()

        if self.data.dataset_plottable is not None:
            self.plot.plot_from_2d(self.data.dataset_plottable)
        else:
            print("There's nothing here! TODO")

    def quicksave_wrapper(self):
        """Save .data directly to default save file."""
        loading.quicksave(self.data)

    def save_to_certain_file_wrapper(self):
        """Open dialog for a file for which to save .data"""
        _ = loading.save_to_user_selected_file(self.data, parent=self)

    def load_file_wrapper(self):
        """Open dialog for file from which to load into .data"""
        self.data = loading.load_by_dialog(parent=self)

        if self.data is not None:
            self.plot.plot_from_2d(self.data.dataset_plottable)
