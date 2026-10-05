import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
from qtpy.QtCore import Qt


class DynamicViewBox(pg.ViewBox):
    """A ViewBox that fits its range to the data on a left double-click."""

    def mouseClickEvent(self, ev: MouseClickEvent) -> None:
        # pyqtgraph has no double-click hook: its scene delivers double clicks here, with ev.double() set.
        if ev.double() and ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self.autoRange()
        else:
            super().mouseClickEvent(ev)  # keeps the right-click context menu
