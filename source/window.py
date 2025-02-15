#!/usr/bin/env python3
import sys
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
)
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QWidget,
)


Const = {
    "WINDOW_TITLE": "Latent Space Visualizer",
    "FILE_FILTERS": [
        "PyTorch Files (*.pt)",
        "All Files (*.*)",
    ],
}


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initiate_menu_bar()

        self.setWindowTitle(Const["WINDOW_TITLE"])

        button = QPushButton("Button")
        self.setCentralWidget(button)


    def initiate_menu_bar(self):
        menu = self.menuBar()
        action_to_open_file = QAction("&Open File", self)
        action_to_open_file.setStatusTip("TODO: Make real status tip")
        action_to_open_file.setCheckable(True)
        action_to_open_file.setShortcut(QKeySequence("Ctrl+O"))


        bogus_menu = menu.addMenu("&Bogus")
        bogus_menu.addAction(action_to_open_file)
        bogus_menu.addAction(QAction("Scrogus", self))


    def get_filename(self):
        initial_filter = FILE_FILTERS[3] # Select one from the list.
        filters = ";;".join(FILE_FILTERS)
        print("Filters are:", filters)
        print("Initial filter:", initial_filter)


class SomeWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("Bottomless Text Field")
        self.setGeometry(300, 300, 300, 150)

        layout = QHBoxLayout()
        self.lineEdit = QLineEdit()
        self.button = QPushButton("Clear")

        layout.addWidget(self.lineEdit)
        layout.addWidget(self.button)

        self.button.pressed.connect(self.lineEdit.clear)

        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
