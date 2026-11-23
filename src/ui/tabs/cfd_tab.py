"""Slide cut planes through a CFD case in 3D: the thinned cases from paraview_scripts/thin_cases.py.

Pick a case and a field, then drag the x or y slider (or use the arrow keys on it) to move the cut plane through the
flow. The model stays still (model frame) and so does the camera: changing slice, field or case never moves the view,
so the same spot can be compared across cases. Each slider keeps its position when the case or field changes.
Left-drag rotates, right-drag or the wheel zooms, middle-drag (or shift + left) pans; "Reset view" goes back to the
view the rendered images used.

Cases in subfolders of the folder are separate sets (pick with "Set"; the cases directly in the folder are "main").
A set of one-actuator-at-a-time cases, e.g. actuator_effects, is browsed as sweeps instead of case by case: pick an
actuator and a flow condition, then drag the deflection slider to watch that surface move through its sweep.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from pyvista.plotting._typing import ScalarBarArgs
from pyvistaqt import QtInteractor
from qtpy.QtCore import Qt
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

from src.cfd import slice_data
from src.cfd.slice_data import CfdCase

CACHED_CASES: int = 16  # loaded cases kept in memory (~20-40 MB each), so stepping back and forth is instant
VIEW_DIRECTION: tuple[float, float, float] = (-0.8, -1.0, 0.6)  # slices.py's "iso" camera
SLICE_LIGHTING: tuple[float, float] = (1.0, 0.0)  # (ambient, diffuse) as slices.py: flat slices in true colours
WALL_LIGHTING: tuple[float, float] = (0.6, 0.4)  # mostly unlit so white reads as white, a little shading for shape
BAR_LABELS: dict[str, int] = {"Cp": 6, "U": 7, "Ux": 5, "Uy": 5, "Uz": 5}  # labels at round steps of each range


def bar_args(field: str, title: str, position_y: float) -> ScalarBarArgs:
    """A vertical colour bar on the right, black text on the white background (as the rendered images)."""
    return {"title": title, "n_labels": BAR_LABELS.get(field, 5), "position_x": 0.9, "position_y": position_y,
            "vertical": True, "color": "black", "fmt": "%.3g", "width": 0.05, "height": 0.35,
            "title_font_size": 14, "label_font_size": 12}


def refill(combo: QComboBox, items: list[str]) -> None:
    """Replace `combo`'s items without signalling, keeping the current text if it is still an item."""
    current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(items)
    if current in items:
        combo.setCurrentText(current)
    combo.blockSignals(False)


class CfdTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._cases: OrderedDict[str, CfdCase] = OrderedDict()  # LRU cache of loaded cases
        self._case: CfdCase | None = None
        self._slice_actors: dict[str, list[pv.Actor]] = {}  # axis -> one actor per slice, only the current shown
        self._camera_set = False

        self._sweeps: slice_data.Sweeps = {}  # the current set's actuator sweeps; empty = pick cases one by one
        self._sweep: list[slice_data.SweepPoint] = []  # the chosen actuator and condition's cases, by deflection

        # --- source row: folder, set, case, field ---
        self.folder_edit = QLineEdit(str(slice_data.CFD_VIEW_DIR))
        self.folder_edit.editingFinished.connect(self.refresh)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        self.set_combo = QComboBox()
        self.set_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.set_combo.currentIndexChanged.connect(lambda _: self._load_set())
        self.case_label = QLabel("Case")
        self.case_combo = QComboBox()
        self.case_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.case_combo.currentTextChanged.connect(lambda _: self._load_case())
        self.field_combo = QComboBox()
        self.field_combo.addItems(list(slice_data.FIELDS))
        self.field_combo.currentTextChanged.connect(lambda _: self._draw())
        reset = QPushButton("Reset view")
        reset.clicked.connect(self._reset_camera)

        source = QHBoxLayout()
        source.addWidget(QLabel("Folder"))
        source.addWidget(self.folder_edit, 1)
        source.addWidget(browse)
        source.addWidget(refresh)
        source.addSpacing(16)
        source.addWidget(QLabel("Set"))
        source.addWidget(self.set_combo)
        source.addWidget(self.case_label)
        source.addWidget(self.case_combo)
        source.addWidget(QLabel("Field"))
        source.addWidget(self.field_combo)
        source.addWidget(reset)

        # --- actuator sweep row (sets of one-actuator cases): actuator, flow condition, deflection slider ---
        self.actuator_combo = QComboBox()
        self.actuator_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.actuator_combo.currentTextChanged.connect(lambda _: self._load_actuator())
        self.condition_combo = QComboBox()
        self.condition_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.condition_combo.currentIndexChanged.connect(lambda _: self._load_condition())
        self.deflection_slider = QSlider(Qt.Orientation.Horizontal)
        self.deflection_slider.setPageStep(1)
        self.deflection_slider.valueChanged.connect(lambda _: self._swept())
        self.deflection_label = QLabel()
        self.deflection_label.setMinimumWidth(130)
        self.sweep_row = QWidget()
        sweep = QHBoxLayout(self.sweep_row)
        sweep.setContentsMargins(0, 0, 0, 0)
        sweep.addWidget(QLabel("Actuator"))
        sweep.addWidget(self.actuator_combo)
        sweep.addWidget(QLabel("Condition"))
        sweep.addWidget(self.condition_combo)
        sweep.addWidget(QLabel("Deflection"))
        sweep.addWidget(self.deflection_slider, 1)
        sweep.addWidget(self.deflection_label)
        self.sweep_row.hide()

        # --- one slider per axis; the radio button says which one is shown ---
        self.axis_group = QButtonGroup(self)
        self.axis_radio: dict[str, QRadioButton] = {}
        self.sliders: dict[str, QSlider] = {}
        self.positions: dict[str, QLabel] = {}
        sliders = QGridLayout()
        for row, axis in enumerate(slice_data.AXES):
            radio = QRadioButton(f"{axis} slices")
            radio.toggled.connect(lambda on: on and self._show())
            self.axis_group.addButton(radio)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setPageStep(1)
            slider.valueChanged.connect(lambda _, a=axis: self._slid(a))
            position = QLabel()
            position.setMinimumWidth(130)
            sliders.addWidget(radio, row, 0)
            sliders.addWidget(slider, row, 1)
            sliders.addWidget(position, row, 2)
            self.axis_radio[axis], self.sliders[axis], self.positions[axis] = radio, slider, position

        # --- 3D view ---
        self.plotter = QtInteractor(self)
        self.plotter.renderer.set_background("white")
        self.plotter.renderer.add_axes()
        self.status = QLabel()
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QVBoxLayout(self)
        layout.addLayout(source)
        layout.addWidget(self.sweep_row)
        layout.addLayout(sliders)
        layout.addWidget(self.plotter.interactor, 1)
        layout.addWidget(self.status)

        self.axis_radio[slice_data.AXES[0]].setChecked(True)
        self.refresh()

    # ----- public -----
    def refresh(self) -> None:
        """Re-read the folder's sets and cases, keeping the selected set and case if they are still there."""
        self._cases.clear()
        current = self.set_combo.currentData()
        sets = slice_data.list_sets(Path(self.folder_edit.text()))
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for name in sets:
            self.set_combo.addItem(name or "main", name)  # the cases directly in the folder are "main"
        if current in sets:
            self.set_combo.setCurrentIndex(sets.index(current))
        self.set_combo.blockSignals(False)
        self._load_set()

    # ----- internals -----
    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Thinned CFD cases folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)
            self.refresh()

    def _set_dir(self) -> Path:
        return Path(self.folder_edit.text()) / (self.set_combo.currentData() or "")

    def _load_set(self) -> None:
        """List the set's cases, and show the actuator sweep row instead of the case list if the set has sweeps."""
        set_dir = self._set_dir()
        current = self.case_combo.currentText()
        cases = slice_data.list_cases(set_dir) if self.set_combo.count() else []
        self.case_combo.blockSignals(True)
        self.case_combo.clear()
        self.case_combo.addItems(cases)
        if current in cases:
            self.case_combo.setCurrentText(current)
        self.case_combo.blockSignals(False)

        self._sweeps = slice_data.actuator_sweeps(set_dir) if cases else {}
        sweeping = bool(self._sweeps)
        self.sweep_row.setVisible(sweeping)
        self.case_label.setVisible(not sweeping)
        self.case_combo.setVisible(not sweeping)
        if sweeping:
            refill(self.actuator_combo, list(self._sweeps))
            self._load_actuator()
        else:
            self._load_case()

    def _load_actuator(self) -> None:
        """The chosen actuator's flow conditions; keeps the condition if this actuator was swept at it too."""
        conditions = list(self._sweeps.get(self.actuator_combo.currentText(), {}))
        current = self.condition_combo.currentData()
        self.condition_combo.blockSignals(True)
        self.condition_combo.clear()
        for alpha, beta in conditions:
            self.condition_combo.addItem(f"AoA {alpha:+.1f}, sideslip {beta:+.1f}", (alpha, beta))
        if current in conditions:
            self.condition_combo.setCurrentIndex(conditions.index(current))
        self.condition_combo.blockSignals(False)
        self._load_condition()

    def _load_condition(self) -> None:
        """The deflections swept at the chosen condition; keeps the deflection if it was swept here too."""
        condition = self.condition_combo.currentData()
        previous = self._sweep[self.deflection_slider.value()].deflection_deg if self._sweep else 0.0
        self._sweep = self._sweeps.get(self.actuator_combo.currentText(), {}).get(condition, [])
        deflections = [p.deflection_deg for p in self._sweep]
        self.deflection_slider.blockSignals(True)
        self.deflection_slider.setRange(0, max(len(self._sweep) - 1, 0))
        if deflections:  # the swept deflection nearest the one shown before
            self.deflection_slider.setValue(min(range(len(deflections)), key=lambda i: abs(deflections[i] - previous)))
        self.deflection_slider.blockSignals(False)
        self.deflection_slider.setEnabled(len(self._sweep) > 1)
        self._swept()

    def _swept(self) -> None:
        """Show the case at the slider's deflection."""
        if not self._sweep:
            self.deflection_label.setText("-")
            return
        i = self.deflection_slider.value()
        self.deflection_label.setText(f"{self._sweep[i].deflection_deg:+.1f} deg   ({i + 1} / {len(self._sweep)})")
        self.case_combo.blockSignals(True)
        self.case_combo.setCurrentText(self._sweep[i].case)
        self.case_combo.blockSignals(False)
        self._load_case()

    def _title(self) -> str:
        """What the view's label calls the current case: the actuator and deflection when sweeping."""
        if self._case is None:
            return ""
        if not self._sweeps or not self._sweep:
            return self._case.name
        point = self._sweep[self.deflection_slider.value()]
        return f"{self.actuator_combo.currentText()} {point.deflection_deg:+.1f} deg   {self.condition_combo.currentText()}"

    def _load_case(self) -> None:
        name = self.case_combo.currentText()
        path = self._set_dir() / name
        key = str(path)
        if not name:
            self._case = None
        elif key in self._cases:
            self._cases.move_to_end(key)
            self._case = self._cases[key]
        else:
            self._case = slice_data.load_case(path)
            self._cases[key] = self._case
            if len(self._cases) > CACHED_CASES:
                self._cases.popitem(last=False)
        for axis in slice_data.AXES:
            n = len(self._case.slices[axis]) if self._case else 0
            slider = self.sliders[axis]
            slider.blockSignals(True)  # keeps its position (clamped) so the same station shows on the new case
            slider.setRange(0, max(n - 1, 0))
            slider.blockSignals(False)
            slider.setEnabled(n > 1)
            self.axis_radio[axis].setEnabled(n > 0)
        self._draw()

    def _draw(self) -> None:
        """Rebuild the scene for the current case and field. The camera is left where it is."""
        camera = self.plotter.camera_position
        self.plotter.clear_actors()
        for title in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title, render=False)
        self._slice_actors = {}
        case, field = self._case, self.field_combo.currentText()
        if case is None or not field:
            self.status.setText("No thinned CFD cases found (see paraview_scripts/thin_cases.py)")
            self.plotter.render()
            return

        same_bar = field == slice_data.WALL_FIELD  # Cp on walls and slices: one map, one range, one bar
        wall_title = "Cp" if same_bar else "Cp (walls)"
        self._add(case, case.walls, slice_data.WALL_FIELD, WALL_LIGHTING, bar_args(slice_data.WALL_FIELD, wall_title, 0.6))
        slice_bar = bar_args(field, slice_data.FIELDS[field].title, 0.1)
        for axis in slice_data.AXES:
            actors: list[pv.Actor] = []
            for s in case.slices[axis]:
                first = not actors and axis == slice_data.AXES[0]  # one bar for all the slices
                actor = self._add(case, s.mesh, field, SLICE_LIGHTING, slice_bar if first and not same_bar else None)
                actor.SetVisibility(False)
                actors.append(actor)
            self._slice_actors[axis] = actors

        if self._camera_set:
            self.plotter.camera_position = camera
        else:
            self._reset_camera()
        self._show()

    def _add(self, case: CfdCase, mesh: pv.PolyData, field: str, lighting: tuple[float, float],
             bar: ScalarBarArgs | None) -> pv.Actor:
        """`mesh` coloured cell by cell by `field` in its fixed colour map and range, with (ambient, diffuse)
        `lighting`, and a colour bar if `bar` is given."""
        ambient, diffuse = lighting
        return self.plotter.add_mesh(
            mesh, scalars=slice_data.field_values(case, mesh, field), preference="cell",
            cmap=ListedColormap(slice_data.colour_table(field)), clim=slice_data.FIELDS[field].range,
            ambient=ambient, diffuse=diffuse, specular=0.0, reset_camera=False,
            show_scalar_bar=bar is not None, scalar_bar_args=bar,
        )

    def _reset_camera(self) -> None:
        """Frame the walls and every slice from slices.py's iso direction, as the rendered images were."""
        case = self._case
        if case is None:
            return
        meshes = [case.walls] + [s.mesh for axis in slice_data.AXES for s in case.slices[axis]]
        bounds = np.array([m.bounds for m in meshes])
        lo, hi = bounds[:, 0::2].min(axis=0), bounds[:, 1::2].max(axis=0)
        centre = (lo + hi) / 2
        direction = np.array(VIEW_DIRECTION) / np.linalg.norm(VIEW_DIRECTION)
        distance = 1.5 * float(np.linalg.norm(hi - lo))
        self.plotter.camera_position = [tuple(centre + distance * direction), tuple(centre), (0.0, 0.0, 1.0)]
        self._camera_set = True
        self.plotter.render()

    def _slid(self, axis: str) -> None:
        if self.axis_radio[axis].isChecked():
            self._show()
        else:
            self.axis_radio[axis].setChecked(True)  # shows via toggled

    def _show(self) -> None:
        """Show only the selected slice."""
        case = self._case
        shown = next(a for a, r in self.axis_radio.items() if r.isChecked())
        for axis in slice_data.AXES:
            slices = case.slices[axis] if case else []
            index = self.sliders[axis].value()
            self.positions[axis].setText(
                f"{index + 1} / {len(slices)}   {axis} = {slices[index].position_mm} mm" if slices else "-")
            for i, actor in enumerate(self._slice_actors.get(axis, [])):
                actor.SetVisibility(axis == shown and i == index)
        if case is not None:
            slices = case.slices[shown]
            where = f"   {shown} = {slices[self.sliders[shown].value()].position_mm} mm" if slices else ""
            self.plotter.add_text(self._title() + where, position="upper_left", font_size=11, color="black",
                                  name="label")
            self.status.setText(str(self._set_dir() / case.name))
        self.plotter.render()
