"""The latest episode replayed on the wing, in the Session tab: the wing from the CAD with every control surface moving
as it was commanded, and the taps coloured by their Cp, on the episode's own clock. The tab's load and command plots
underneath follow it with a cursor (`time_shown`).

It shows the episode the actor last wrote, so it runs one episode (about horizon * dt + settle) behind the rig, and
loops until the next one arrives. Display only: the geometry, the hinge transforms and the tap positions come from
src/cfd/wing_view.py, the time series from the SessionReader.
"""

import time
from typing import Literal

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from src.cfd import slice_data
from src.cfd.wing_view import WingModel, load_wing_model, tap_positions
from src.ml.distributed.session import EpisodeDetail

type TapsMode = Literal["fluctuation", "absolute"]
type CommandMode = Literal["applied", "requested"]

FRAME_MS = 33  # ~30 frames a second
TAP_POINT_SIZE = 10.0  # pixels
CAMERA_ZOOM = 1.5  # the CFD tab's iso view, moved in: the wing alone, no flow slices around it


class ReplayPanel(QWidget):
    time_shown = Signal(float)  # the episode time [s] of the frame on screen

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        self.title = QLabel("Wing: waiting for an episode")
        self.play_btn = QPushButton("Pause")
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self._scrub)

        controls = QHBoxLayout()
        controls.addWidget(self.title, 1)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.slider, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.plotter, 1)

        self._model: WingModel | None = None
        self._surface_actors: dict[str, pv.Actor] = {}
        self._taps_mesh: pv.PolyData | None = None
        self._detail: EpisodeDetail | None = None
        self._taps_mode: TapsMode = "fluctuation"
        self._command_mode: CommandMode = "applied"
        self._t = np.zeros(0)
        self._deg: np.ndarray | None = None
        self._taps: np.ndarray | None = None
        self._frame = 0
        self._playing = True
        self._clock = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(FRAME_MS)

    # ----- data -----
    def show_episode(self, detail: EpisodeDetail) -> None:
        """Replay `detail` from its start, unless it is the episode already shown."""
        if self._detail is not None and self._detail.meta.episode_id == detail.meta.episode_id:
            return
        if not self._ensure_scene():
            return
        self._detail = detail
        self._t = detail.t.numpy()
        self._pick_deflections()
        self._colour_taps()
        self.slider.setRange(0, max(len(self._t) - 1, 0))
        kind = "eval" if detail.meta.evaluate else "baseline" if detail.meta.baseline else "train"
        self.title.setText(f"Episode #{detail.meta.episode_id} on the wing ({kind}, weights v{detail.meta.policy_version}), "
                           "looping until the next one")
        self._clock = time.perf_counter()
        self._show_frame(0)

    def set_taps_mode(self, mode: TapsMode) -> None:
        """fluctuation: each tap's Cp minus its episode mean, so the gusts are not hidden under the steady
        distribution; absolute: Cp as measured."""
        self._taps_mode = mode
        self._colour_taps()
        self._show_frame(self._frame)

    def set_command_mode(self, mode: CommandMode) -> None:
        """applied: where the surfaces were sent, after the safety layer's rate limit and clamp; requested: where the
        policy asked for them."""
        self._command_mode = mode
        self._pick_deflections()
        self._show_frame(self._frame)

    def _pick_deflections(self) -> None:
        d = self._detail
        if d is None:
            self._deg = None
            return
        deg = d.requested_deg if self._command_mode == "requested" or d.deflection_deg is None else d.deflection_deg
        self._deg = None if deg is None else deg.numpy()

    def _colour_taps(self) -> None:
        d = self._detail
        n_taps = 0 if self._taps_mesh is None else self._taps_mesh.n_points
        if d is None or d.taps is None or d.taps.shape[1] != n_taps:
            self._taps = None
            return
        taps = d.taps.numpy()
        if self._taps_mode == "fluctuation":
            taps = taps - taps.mean(axis=0, keepdims=True)
            span = float(np.percentile(np.abs(taps), 99)) or 1.0  # robust to one noisy spike
            levels = (-span, span)
        else:
            centre = float(np.median(taps))
            span = float(np.percentile(np.abs(taps - centre), 99)) or 1.0
            levels = (centre - span, centre + span)
        self._taps = taps
        self.plotter.update_scalar_bar_range(list(levels))
        self.plotter.scalar_bar.SetTitle("Cp - mean" if self._taps_mode == "fluctuation" else "Cp")

    def _ensure_scene(self) -> bool:
        """Load the wing and build the scene, once. False (with the reason shown) if it cannot be loaded."""
        if self._model is not None:
            return True
        try:
            model = load_wing_model()
            taps = tap_positions()
        except (OSError, ValueError) as e:
            self.title.setText(f"Wing unavailable: {e}")
            return False
        self._model = model
        for mesh in model.fixed.values():
            self.plotter.add_mesh(mesh, color="#b0b0b0", smooth_shading=True)
        for name, mesh in model.surfaces.items():
            self._surface_actors[name] = self.plotter.add_mesh(mesh, color="#5b8fd6", smooth_shading=True)
        self._taps_mesh = pv.PolyData(taps)
        self._taps_mesh.point_data["Cp"] = np.zeros(len(taps))
        self.plotter.add_mesh(self._taps_mesh, scalars="Cp", cmap="RdBu", render_points_as_spheres=True,
                              point_size=TAP_POINT_SIZE,
                              scalar_bar_args={"title": "Cp", "n_labels": 3, "fmt": "%.2f", "vertical": True})
        self.plotter.camera_position = slice_data.iso_camera([*model.fixed.values(), *model.surfaces.values()],
                                                             zoom=CAMERA_ZOOM)
        return True

    # ----- playback -----
    def _tick(self) -> None:
        if not self._playing or not len(self._t) or not self.isVisible():
            return
        elapsed = (time.perf_counter() - self._clock) % float(self._t[-1])  # loop in real time
        self._show_frame(int(np.searchsorted(self._t, elapsed)))

    def _show_frame(self, i: int) -> None:
        if not len(self._t):
            return
        i = min(max(i, 0), len(self._t) - 1)
        self._frame = i
        model = self._model
        if model is not None and self._deg is not None:
            for k, name in enumerate(model.rig_order):
                self._surface_actors[name].user_matrix = model.surface_matrix(name, float(self._deg[i, k]))
        if self._taps is not None and self._taps_mesh is not None:
            self._taps_mesh.point_data["Cp"] = self._taps[i]
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.plotter.render()
        self.time_shown.emit(float(self._t[i]))

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self.play_btn.setText("Pause" if self._playing else "Play")
        if self._playing and len(self._t):  # carry on from the frame shown
            self._clock = time.perf_counter() - float(self._t[self._frame])

    def _scrub(self, i: int) -> None:
        if self._playing:
            self._toggle_play()
        self._show_frame(i)

    def close_scene(self) -> None:
        self._timer.stop()
        self.plotter.close()
