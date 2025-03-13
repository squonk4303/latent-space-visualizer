#!/usr/bin/env python3
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

from visualizer import consts
from visualizer import loading
from visualizer.plot_widget import PlotWidget
from visualizer.stacked_layout_manager import StackedLayoutManager


class PrimaryWindow(QMainWindow):
    """
    Primary window of the program
    This is where all the action happens
    """

    def __init__(self, *args, **kwargs):
        """Constructor for the primary window"""
        super().__init__(*args, **kwargs)

        # Set up the tabs in the window
        start_tab = QVBoxLayout()
        graph_tab = QVBoxLayout()

        self.tab_layout = StackedLayoutManager()
        self.tab_layout.add_layout(start_tab)
        self.tab_layout.add_layout(graph_tab)

        # Add buttons to navigate to each tab
        self.start_tab_button = QPushButton("0")
        self.graph_tab_button = QPushButton("1")
        self.start_tab_button.clicked.connect(self.signal_tab(0))
        self.graph_tab_button.clicked.connect(self.signal_tab(1))

        tab_buttons_layout = QHBoxLayout()
        tab_buttons_layout.addWidget(self.start_tab_button)
        tab_buttons_layout.addWidget(self.graph_tab_button)

        # And put them in order
        greater_layout = QVBoxLayout()
        greater_layout.addLayout(tab_buttons_layout)
        greater_layout.addLayout(self.tab_layout)

        # Set up the menu bar and submenus
        self.initiate_menu_bar()

        # --- Initialize start screen ---

        # Row For Model Selection
        # -----------------------
        self.model_feedback_label = QLabel("Choose a model already...")
        openfile_button = QPushButton(consts.OPEN_FILE_LABEL)
        openfile_button.clicked.connect(self.load_model_file)

        row_model_selection = QHBoxLayout()
        row_model_selection.addWidget(openfile_button)
        row_model_selection.addWidget(self.model_feedback_label)

        # Row For Category Selection
        # --------------------------
        self.category_feedback_label = QLabel("Give me skin...")
        category_button = QPushButton("Select Categories")

        row_category_selection = QHBoxLayout()
        row_category_selection.addWidget(category_button)
        row_category_selection.addWidget(self.category_feedback_label)

        # Row For Layer Selection
        # -----------------------
        self.layer_feedback_label = QLabel(
            "Are you going to just stand there or will you select a layer??"
        )
        layer_button = QPushButton("Select layer")

        row_layer_selection = QHBoxLayout()
        row_layer_selection.addWidget(layer_button)
        row_layer_selection.addWidget(self.layer_feedback_label)

        # Row For Dataset Selection
        # -------------------------
        dataset_selection_button = QPushButton("Open dat shit...")
        self.dataset_feedback_label = QLabel("Select a dataset, yo...")
        dataset_selection_button.clicked.connect(self.find_dataset)

        row_dataset_selection = QHBoxLayout()
        row_dataset_selection.addWidget(dataset_selection_button)
        row_dataset_selection.addWidget(self.dataset_feedback_label)

        # Row For Single Picture Selection
        # --------------------------------
        # Note on pixmap from https://doc.qt.io/qt-6/qpixmap.html#details
        # QPixmap is designed and optimized for showing images on screen
        self.single_image_label = QLabel("Hoping for a chicken.")
        self.single_image_thumb_label = QLabel()
        self.single_image_thumb_label.setPixmap(QPixmap("assets/default_pic.png"))
        single_image_button = QPushButton("Open ...Image")
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
        self.plot = PlotWidget()
        toolbar = self.plot.make_toolbar()

        graph_tab.addWidget(self.plot)
        graph_tab.addWidget(toolbar)

        # --- Window Configuration ---
        self.resize(650, 450)
        self.setWindowTitle(consts.WINDOW_TITLE)
        self.setStatusBar(QStatusBar(self))
        widget = QWidget()
        widget.setLayout(greater_layout)
        self.setCentralWidget(widget)

    def initiate_menu_bar(self):
        """Set up the top menu-bar, its sub-menues, actions, and signals."""
        menubar = self.menuBar()

        # Action which opens the file dialog
        action_to_open_file = QAction(consts.OPEN_FILE_LABEL, self)
        action_to_open_file.triggered.connect(self.load_model_file)

        # Actions to scroll to next/previous tabs
        next_tab = QAction("TEMP: &Next tab")
        prev_tab = QAction("TEMP: &Previous tab")
        # https://doc.qt.io/qt-6/qkeysequence.html#StandardKey-enum
        next_tab.setShortcut(QKeySequence.StandardKey.MoveToNextPage)
        prev_tab.setShortcut(QKeySequence.StandardKey.MoveToPreviousPage)

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
        # If user cancels dialog, does nothing
        if model_path:
            # @Wilhelmsen: Test for this
            # @Wilhelmsen: And make it more displayable before release
            # @Wilhelmsen: And store it in at attribute or something so it's usable
            self.model_feedback_label.setText("You chose: " + model_path)

    def find_dataset(self):
        """
        Called by dataset_selection_button.
        @Wilhelmsen, do this at some point
        """
        # @Wilhelmsen: The FDM should really be a singleton you know...
        handler = loading.FileDialogManager(self)
        dataset_dir = handler.find_directory()
        if dataset_dir:
            self.dataset_feedback_label.setText("You found: " + dataset_dir)

    def find_picture(self):
        """."""
        handler = loading.FileDialogManager(self)
        picture_path = handler.find_picture_file()
        if picture_path:
            # @Wilhelmsen: Yet to check validity and resize image
            self.single_image_label.setText(picture_path)
            self.single_image_thumb_label.setPixmap(QPixmap(picture_path))

            # Start the process of dim.reducing the image
            big_obj = loading.AutoencodeModel()
            tensor = big_obj.single_image_to_tensor(picture_path)
            print(tensor)
            # And take it through t-SNE just for good measure too
            # Or not...
            # Maybe it's best to leave that for whwn things are plotted onto the graph...
            # I'm wondering if t-SNED coords shouldn't be stored over there anyways.
            # Then again it could be helpful to have them put over there immediately...

    def start_cooking(self):
        # loaded_model = loading.get_model(model_path, categories)
        # loading.layer_summary(loaded_model, 1, 2)
        # reduced_data = loading.reduce_data(model_path, categories)

        big_obj = loading.AutoencodeModel()
        # @Wilhelmsen: The way the model dict works is ridiculous. Change it in the refactoring process.
        big_obj.load_model("no_augmentation", model_path, categories)
        big_obj.the_whole_enchilada("no_augmentation")

        reduced_data = loading.the_whole_enchilada()
        print(reduced_data)
        self.plot.plot_from_2d(reduced_data)

    def signal_tab(self, n):
        def func():
            self.tab_layout.setCurrentIndex(n)

        return func
