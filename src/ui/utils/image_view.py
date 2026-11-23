"""An image viewer with mouse-wheel zoom and drag-to-pan that keeps its zoom and pan when the image is swapped."""

from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent, QPainter, QPixmap, QResizeEvent, QWheelEvent
from qtpy.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

ZOOM_STEP: float = 1.25  # per wheel notch
MAX_ZOOM: float = 40.0  # relative to fitting the whole image


class ImageView(QGraphicsView):
    """Wheel zooms about the cursor, left-drag pans, double-click (or `fit`) shows the whole image again.

    While fitted, the image follows the widget's size. Once zoomed in, `set_pixmap` keeps the view where it is, so
    flicking through images of the same size compares the same spot."""

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._item)
        self._fitted = True
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumSize(200, 120)

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        """Show `pixmap` (None clears). Keeps the current zoom and pan unless the view is fitted."""
        new_size = pixmap is not None and pixmap.size() != self._item.pixmap().size()
        self._item.setPixmap(pixmap if pixmap is not None else QPixmap())
        self.scene().setSceneRect(self._item.boundingRect())
        if self._fitted or new_size:
            self.fit()

    def fit(self) -> None:
        """Show the whole image."""
        self._fitted = True
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        pixmap = self._item.pixmap()
        if pixmap.isNull():
            return
        viewport = self.viewport().size()
        fit_scale = min(viewport.width() / pixmap.width(), viewport.height() / pixmap.height())
        scale = self.transform().m11()
        target = scale * ZOOM_STEP ** (event.angleDelta().y() / 120)
        if target <= fit_scale:
            self.fit()
            return
        target = min(target, MAX_ZOOM * fit_scale)
        self.scale(target / scale, target / scale)  # about the cursor (AnchorUnderMouse)
        self._fitted = False

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.fit()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fitted:
            self.fit()
