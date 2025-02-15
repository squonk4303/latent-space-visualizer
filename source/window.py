#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLineEdit, QPushButton, QHBoxLayout, QMainWindow,
)


Const = {
    "WINDOW_TITLE" : "Latent Space Visualizer",
}


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle(const["WINDOW_TITLE"])

        button = QPushButton("Button")
        self.setCentralWidget(button)



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
