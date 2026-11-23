"""Slide cut planes through a CFD case: flick through the slice images rendered by slices.py.

Pick a case and a field, then drag the x or y slider (or use the arrow keys on it). The model and camera are the same
in every image, so it looks like moving a slice through the flow. Each slider keeps its position when the case or
field changes, so the same station can be compared across cases. Mouse wheel zooms, dragging pans and a double-click
fits the whole image; the zoom and pan stay put while sliding, so one region can be followed through the sweep.
"""

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.cfd import slice_images
from src.ui.utils.image_view import ImageView


class CfdTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._sweeps: dict[str, list[Path]] = {}  # axis -> images of the current case and field
        self._pixmaps: dict[Path, QPixmap] = {}  # loaded images of the current case

        # --- source row: folder, case, field ---
        self.folder_edit = QLineEdit(str(slice_images.SLICE_IMAGES_DIR))
        self.folder_edit.editingFinished.connect(self.refresh)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        self.case_combo = QComboBox()
        self.case_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.case_combo.currentTextChanged.connect(lambda _: self._load_case())
        self.field_combo = QComboBox()
        self.field_combo.currentTextChanged.connect(lambda _: self._load_sweeps())

        source = QHBoxLayout()
        source.addWidget(QLabel("Folder"))
        source.addWidget(self.folder_edit, 1)
        source.addWidget(browse)
        source.addWidget(refresh)
        source.addSpacing(16)
        source.addWidget(QLabel("Case"))
        source.addWidget(self.case_combo)
        source.addWidget(QLabel("Field"))
        source.addWidget(self.field_combo)

        # --- one slider per axis; the radio button says which one is shown ---
        self.axis_group = QButtonGroup(self)
        self.axis_radio: dict[str, QRadioButton] = {}
        self.sliders: dict[str, QSlider] = {}
        self.positions: dict[str, QLabel] = {}
        sliders = QGridLayout()
        for row, axis in enumerate(slice_images.AXES):
            radio = QRadioButton(f"{axis} slices")
            radio.toggled.connect(lambda on: on and self._show())
            self.axis_group.addButton(radio)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setPageStep(1)
            slider.valueChanged.connect(lambda _, a=axis: self._slid(a))
            position = QLabel()
            position.setMinimumWidth(60)
            sliders.addWidget(radio, row, 0)
            sliders.addWidget(slider, row, 1)
            sliders.addWidget(position, row, 2)
            self.axis_radio[axis], self.sliders[axis], self.positions[axis] = radio, slider, position

        # --- image: wheel to zoom, drag to pan, double-click to fit; zoom and pan stay put while sliding ---
        self.image = ImageView()
        self.path_label = QLabel()
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QVBoxLayout(self)
        layout.addLayout(source)
        layout.addLayout(sliders)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.path_label)

        self.axis_radio[slice_images.AXES[0]].setChecked(True)
        self.refresh()

    # ----- public -----
    def refresh(self) -> None:
        """Re-read the folder's cases, keeping the selected case if it is still there."""
        current = self.case_combo.currentText()
        cases = slice_images.list_cases(Path(self.folder_edit.text()))
        self.case_combo.blockSignals(True)
        self.case_combo.clear()
        self.case_combo.addItems(cases)
        if current in cases:
            self.case_combo.setCurrentText(current)
        self.case_combo.blockSignals(False)
        self._load_case()

    # ----- internals -----
    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Slice images folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)
            self.refresh()

    def _case_dir(self) -> Path:
        return Path(self.folder_edit.text()) / self.case_combo.currentText()

    def _load_case(self) -> None:
        self._pixmaps.clear()
        current = self.field_combo.currentText()
        fields = slice_images.list_fields(self._case_dir()) if self.case_combo.currentText() else []
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(fields)
        if current in fields:
            self.field_combo.setCurrentText(current)
        self.field_combo.blockSignals(False)
        self._load_sweeps()

    def _load_sweeps(self) -> None:
        field = self.field_combo.currentText()
        for axis in slice_images.AXES:
            images = slice_images.sweep(self._case_dir(), field, axis) if field else []
            self._sweeps[axis] = images
            slider = self.sliders[axis]
            slider.blockSignals(True)  # keeps its position (clamped) so the same station shows on the new sweep
            slider.setRange(0, max(len(images) - 1, 0))
            slider.blockSignals(False)
            slider.setEnabled(len(images) > 1)
            self.axis_radio[axis].setEnabled(bool(images))
        self._show()

    def _slid(self, axis: str) -> None:
        if self.axis_radio[axis].isChecked():
            self._show()
        else:
            self.axis_radio[axis].setChecked(True)  # shows via toggled

    def _show(self) -> None:
        for axis in slice_images.AXES:
            n = len(self._sweeps.get(axis, []))
            self.positions[axis].setText(f"{self.sliders[axis].value() + 1} / {n}" if n else "-")
        axis = next(a for a, r in self.axis_radio.items() if r.isChecked())
        images = self._sweeps.get(axis, [])
        if not images:
            self.image.set_pixmap(None)
            self.path_label.setText("No slice images found")
            return
        path = images[self.sliders[axis].value()]
        if path not in self._pixmaps:
            self._pixmaps[path] = QPixmap(str(path))
        self.image.set_pixmap(self._pixmaps[path])
        self.path_label.setText(str(path))
