#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLineEdit, QPushButton, QHBoxLayout,
)


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
    window = SomeWindow()

    window.show()
    app.exec()
