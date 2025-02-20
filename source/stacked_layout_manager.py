#!/usr/bin/env python3
from PyQt6.QtWidgets import (
    QStackedLayout,
    QWidget,
)


class StackedLayoutManager(QStackedLayout):
    """
    Class to handle the layout
    It's like a data structure I made
    TODO: method to remove items
    TODO: See about doing some overload to make the items part of the class
          itself. Ex: mystackedlayout[2]; self.append(QWidget()); # et.c.
    """
    def __init__(self, items=None):
        """ Creates an empty stacked layout """
        super().__init__()
        # Casts to list() in case 'items' is a tuple or something
        # TODO: Include an error check here
        self.items = list() if items is None else list(items)
        self.setCurrentIndex(len(self.items))
        self.setCurrentIndex(400)

    def add_widget(self, widget):
        """ Appends a widget to the layout """
        self.addWidget(widget)
        self.items.append(widget)

    def add_layout(self, qlayout):
        """ Appends a layout to the layout """
        tab = QWidget()
        tab.setLayout(qlayout)
        self.addWidget(tab)
        self.items.append(self)

    def scroll_somewhere(self, n=1):
        """ Scrolls to a layer relatively , according to n """
        if n <= 0:
            pass  # TODO: Implement error message
        maximum = self.count()
        current_index = self.currentIndex()
        new_index = (current_index + int(n)) % maximum
        self.setCurrentIndex(new_index)

    def scroll_forth(self):
        """ Scrolls to next layer """
        self.scroll_somewhere(1)

    def scroll_back(self):
        """ Scrolls to prev layer """
        self.scroll_somewhere(-1)
