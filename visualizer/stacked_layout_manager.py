#!/usr/bin/env python3
from PyQt6.QtWidgets import (
    QStackedLayout,
    QWidget,
)


class StackedLayoutManager(QStackedLayout):
    """Class to manage a stacked layout to replicate a tabbed interface."""

    def __init__(self, items=None):
        """Create an empty stacked layout."""
        super().__init__()
        if items is not None:
            for item in items:
                self.add_widget(item)

    def add_widget(self, widget):
        """Append a widget to the layout."""
        self.addWidget(widget)

    def add_layout(self, layout):
        """Append a layout to the layout."""
        widget = QWidget()
        widget.setLayout(layout)
        self.addWidget(widget)

    def scroll_somewhere(self, n=1):
        """Scroll to a layer relatively , according to n."""
        if self.count() != 0:
            maximum = self.count()
            current_index = self.currentIndex()
            new_index = (current_index + int(n)) % maximum
            self.setCurrentIndex(new_index)

    def scroll_forth(self):
        """Scroll to next layer."""
        self.scroll_somewhere(1)

    def scroll_back(self):
        """Scroll to prev layer."""
        self.scroll_somewhere(-1)
