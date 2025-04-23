#!/usr/bin/env python3
"""
Module with functions tohandle file dialogs.
"""
import datetime
import re
import mmap
import tempfile

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from visualizer import consts


def for_file(
    caption="",
    *,
    parent,
    file_filters=consts.FILE_FILTERS.values(),
    options=QFileDialog.Option.ReadOnly,
):
    """
    Launch a file dialog and return the filepath and selected filter.

    Note that the first element in file_filters is used as the initial file filter.
    """
    # Test whether file_filters is a subset of-- or equal to, the legal file filters
    if not set(file_filters) <= set(consts.FILE_FILTERS.values()):
        raise RuntimeError("Unacceptable list of file filters")

    # Generate Qt-readable filter specifications
    file_filters = list(file_filters)
    initial_filter = file_filters[0]
    filters = ";;".join(file_filters)

    # This function opens a nifty Qt-made file dialog
    filepath, selected_filter = QFileDialog.getOpenFileName(
        parent=parent,
        caption=caption,
        filter=filters,
        initialFilter=initial_filter,
        options=options,
    )

    return filepath, selected_filter


def to_save_file(
    caption="",
    *,
    parent,
    file_filters=consts.FILE_FILTERS.values(),
    options=QFileDialog.Option.ReadOnly,
):
    """Launch file dialog where user is prompted for where to save a .pickle file."""
    # Test whether file_filters is a subset of-- or equal to, the legal file filters
    if not set(file_filters) <= set(consts.FILE_FILTERS.values()):
        raise RuntimeError("Unacceptable list of file filters")

    # Generate Qt-readable filter specifications
    file_filters = list(file_filters)
    initial_filter = file_filters[0]
    filters = ";;".join(file_filters)

    # Make default filename based on date/time
    default_filename = datetime.datetime.now().strftime(
        "save_data/%y%m%d-%H%M%S-data.pickle"
    )

    # This function opens a nifty Qt-made file dialog
    filepath, selected_filter = QFileDialog.getSaveFileName(
        parent=parent,
        caption=caption,
        directory=default_filename,
        filter=filters,
        initialFilter=initial_filter,
        options=options,
    )

    return filepath, selected_filter


def for_some_file(caption="", *, parent):
    """Launch a file dialog where user is prompted to pick out any file their heart desires."""
    filepath, _ = for_file(caption, parent=parent)
    return filepath


def for_image_file(caption=consts.IMAGE_DIALOG_CAPTION, *, parent):
    """Launch file dialog where user is intended to pick out a graphical image file."""
    filters = [
        consts.FILE_FILTERS["pictures"],
        consts.FILE_FILTERS["whatever"],
    ]
    filepath, _ = for_file(caption, parent=parent, file_filters=filters)
    return filepath


def for_trained_model_file(caption=consts.TRAINED_MODEL_DIALOG_CAPTION, *, parent):
    """Launch file dialog where user is intended to pick out the file for a trained nn-model."""
    filters = [
        consts.FILE_FILTERS["pytorch"],
        consts.FILE_FILTERS["whatever"],
    ]
    filepath, _ = for_file(caption, parent=parent, file_filters=filters)
    return filepath


def for_directory(caption="", *, parent):
    """Launch a file dialog where user is prompted to pick out a directory."""
    dirpath = QFileDialog.getExistingDirectory(
        parent=parent,
        caption=caption,
    )
    return dirpath


def for_layer_select(model, caption="", *, parent):
    startlayer, endlayer = LayerDialog().get_layers(model=model, caption=caption, parent=parent)
    print("After dialog, got layer", startlayer, endlayer)
    return startlayer, endlayer


class LayerDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.layer_pattern = re.compile(r"layer\d+\.*\d*")
        self.number_pattern = re.compile(r"\d+\.*\d*")

        # Widgets and gidgets
        left_label = QLabel("Start Layer:")
        self.startButton = QComboBox(parent=self)
        self.startButton.addItem("...")
        self.startButton.currentTextChanged.connect(self.startbox_changed)
        left_col = QVBoxLayout()
        left_col.addWidget(left_label)
        left_col.addWidget(self.startButton)

        right_label = QLabel("End Layer:")
        self.endButton = QComboBox(parent=self)
        self.endButton.addItem("...")
        self.endButton.currentTextChanged.connect(self.endbox_changed)
        right_col = QVBoxLayout()
        right_col.addWidget(right_label)
        right_col.addWidget(self.endButton)

        # Set the submit button to simply close the window
        submitButton = QPushButton("Submit")
        submitButton.clicked.connect(self.done)

        layout = QVBoxLayout()
        self.textbox = QTextEdit()
        self.textbox.setReadOnly(True)
        self.textbox.setPlainText("...")
        subLayout = QHBoxLayout()

        subLayout.addLayout(left_col)
        subLayout.addLayout(right_col)

        layout.addWidget(self.textbox)
        layout.addLayout(subLayout)
        layout.addWidget(submitButton)

        self.setLayout(layout)

    def expand_buttons(self, layers):
        self.startButton.clear()
        self.endButton.clear()
        self.startButton.addItem("...")
        self.endButton.addItem("...")
        self.startButton.addItems(layers)
        self.endButton.addItems(layers)

    def startbox_changed(self, text: str):
        """
        Finds "layer*" from selected item and rewrites the textbox accordingly

        Is called every time the start dropdown menu is invoked with a new value.
        """
        # Really just finds IF the user selected a layer start
        layer_match = self.layer_pattern.search(text)

        if layer_match:
            print(layer_match.group())
            # Find what number (with decimals) was found
            number_match = self.number_pattern.search(text)
            self.start_input = int(number_match.group())
            print("Start/end input:", self.start_input, self.end_input)
            # @Wilhelmsen: textbox doesn't seem to take for some reason
            # @Linnea: Any idea?
            # Set textbox again from the stuff found
            self.paramdict_lines = self.layer_summary(
                self.model, self.start_input, self.end_input
            )
            self.textbox.setPlainText("".join(self.paramdict_lines))

    def endbox_changed(self, text: str):
        """
        Finds "layer*" from selected item and rewrites the textbox accordingly

        Is called every time the start dropdown menu is invoked with a new value.
        """
        # Really just finds IF the user selected a layer start
        layer_match = self.layer_pattern.search(text)

        if layer_match:
            print(layer_match.group())
            # Find what number (with decimals) was found
            number_match = self.number_pattern.search(text)
            self.end_input = int(number_match.group())
            print("Start/end input:", self.start_input, self.end_input)
            # @Wilhelmsen: textbox doesn't seem to take for some reason
            # @Linnea: Any idea?
            # Set textbox again from the stuff found
            self.paramdict_lines = self.layer_summary(
                self.model, self.start_input, self.end_input
            )
            self.textbox.setPlainText("".join(self.paramdict_lines))

    def get_layers(self, model, caption="Layer Dialog", *, parent):
        # self.setParent(parent)  <<< TODO; has a weird effect
        self.setWindowTitle(caption)
        self.resize(650, 450)
        self.model = model
        self.start_input = 0
        self.end_input = 0

        self.paramdict_lines = self.layer_summary(model)
        # self.textbox.setPlainText(str(model))
        self.textbox.setPlainText("".join(self.paramdict_lines))
        self.expand_buttons(self.layer_menu_maker(self.paramdict_lines))
        self.exec()

        # @Linnea: Feel free to make this something more sensible
        # Returns both start and end layers
        if self.start_input == 0:
            return None, None
        else:
            return "layer" + str(self.start_input), "layer" + str(self.end_input)

    def layer_menu_maker(self, list_):
        menu_of_layers =[]
        for line in list_:
            match = self.layer_pattern.search(line)
            if match is not None:
                menu_of_layers.append(match.group())
        return menu_of_layers

    def layer_summary(self, loaded_model, start_layer=0, end_layer=0):
        """
        Summarises selected layers from a given model objet.
        If endlayer is left blank only return one layer.
        If start layer is left blank returns all layers.
        If both layers are specified returns from startlayer up to
        and including the endlayer!
        """
        # Sets basic logic and variables
        all_layers = False
        if not end_layer:
            end_layer = start_layer
        if not start_layer:
            all_layers = True

        input_txt = str(loaded_model)
        target = "layer"
        # Assigns targetlayers for use in search later
        next_layer = target + str(end_layer + 1)
        target += str(start_layer)

        """
        At some point in this function an extraction function is to be added
        to filter the information and only return the useful information and attributes
        to be added to the list. For now it takes the entire line of information.
        """

        # Create a temporary data file to store data in a list
        lines = []
        with tempfile.TemporaryFile("wb+", 0) as file:
            file.write(input_txt.encode("utf-8"))
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            while True:
                byteline = mm.readline()
                if byteline:
                    lines.append(byteline.decode("utf-8"))
                else:
                    break
            mm.close()

        output = []

        # Returns selected layers
        found = False
        eol = False
        new = 0
        for i, line in enumerate(lines):
            if all_layers:
                pass
            elif target in line:
                found = True
            elif next_layer in line:
                eol = True
                new = i
            if all_layers or found and not eol:
                output.append(str(f"{i}: {line}"))

        # End of print
        if all_layers:
            output.append(str("\nEOF: no more lines"))
        else:
            output.append(str(f"\nNext line is {new}: {lines[new]}"))
        
        return output
