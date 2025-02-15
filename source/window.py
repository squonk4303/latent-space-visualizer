#!/usr/bin/env python3
import sys
from consts import Const
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QPushButton,
)


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initiate_menu_bar()

        self.open_file_button = QPushButton(Const.OPEN_FILE_LABEL)
        self.open_file_button.clicked.connect(self.get_filename)
        # TODO: Should this be an action?     ^^^^^^^^^^^^^^^^^

        self.setWindowTitle(Const.WINDOW_TITLE)
        self.setCentralWidget(self.open_file_button)

    def initiate_menu_bar(self):
        menu = self.menuBar()
        self.action_to_open_file = QAction("&Open File", self)
        self.action_to_open_file.setStatusTip(Const.STATUS_TIP_TEMP)
        self.action_to_open_file.setShortcut(QKeySequence("Ctrl+O"))
        self.action_to_open_file.triggered.connect(self.get_filename)
        # Note the function Raference              ^^^^^^^^^^^^^^^^^

        self.file_menu = menu.addMenu("&File")
        self.file_menu.addAction(self.action_to_open_file)

    def get_filename(self):
        initial_filter = Const.FILE_FILTERS[0]  # Select one from the list.
        filters = ";;".join(Const.FILE_FILTERS)
        print("Filters are:", filters)
        print("Initial filter:", initial_filter)

        filename, selected_filter = QFileDialog.getOpenFileName(
            self,
            filter=filters,
            initialFilter=initial_filter,
        )
        print("Result:", filename, selected_filter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
