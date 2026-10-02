import numpy as np
import pyqtgraph as pg
from airfoil import Airfoil, Surface, SurfaceData, naca4, placeholder_cp, surface_data
from numpy.typing import NDArray
from pyqtgraph import PlotWidget
from qtpy.QtCore import Qt
from qtpy.QtGui import QCloseEvent, QColor, QPalette
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from training import Progress, SurfaceProgress, TrainWorker

pg.setConfigOptions(antialias=True)

Array = NDArray[np.float64]

# Fixed plot limits so the view never rescales when parameters change
X_RANGE = (-0.02, 1.02)
AIRFOIL_Y_RANGE = (-0.3, 0.3)
PRESSURE_Y_RANGE = (-3.0, 2.5)

BLUE = "#5e9bff"
RED = "#ff6b6b"
TAP_COLOR = "#ffb703"

GRID_POINTS = 400  # resolution of the GP curve drawn over the surface


def with_alpha(color: QColor, alpha: int) -> QColor:
    result = QColor(color)
    result.setAlpha(alpha)
    return result


def tap_style(final: bool, outline: QColor) -> dict[str, object]:
    """Inducing points are drawn faint while training and solid once final."""
    return {
        "symbol": "d",
        "size": 13 if final else 9,
        "brush": pg.mkBrush(with_alpha(QColor(TAP_COLOR), 255 if final else 140)),
        "pen": pg.mkPen(outline, width=1),
    }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pressure Tap GP")
        self.resize(1200, 750)

        self._worker: TrainWorker | None = None
        self._run_id = 0  # bumped on every start/cancel so stale worker signals are ignored
        self._surface: SurfaceData | None = None
        self._grids: tuple[Array, Array] = (np.empty(0), np.empty(0))  # arc-length grids (upper, lower)
        self._last_progress: Progress | None = None

        self.init_ui()
        self.update_airfoil_plot(self.airfoil_plot, 0, 5, 0.12)  # Initial plot with default values

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        plots_layout = QVBoxLayout()
        side_layout = QVBoxLayout()
        layout.addLayout(plots_layout, stretch=3)
        layout.addLayout(side_layout, stretch=2)

        self.airfoil_plot = AirfoilPlot()
        self.pressure_plot = PressurePlot()
        self.pressure_plot.setXLink(self.airfoil_plot)  # keep the chordwise axes aligned
        plots_layout.addWidget(self.airfoil_plot, stretch=1)
        plots_layout.addWidget(self.pressure_plot, stretch=2)

        self.training_plot = TrainingPlot()
        side_layout.addWidget(self.training_plot, stretch=1)

        control_layout = QVBoxLayout()
        camber_label = QLabel("Camber (% chord):")
        camber_input = QDoubleSpinBox()
        camber_input.setRange(0, 10)
        camber_input.setSingleStep(0.5)
        camber_input.setValue(0)
        camber_input.valueChanged.connect(lambda value: self.update_airfoil_plot(self.airfoil_plot, value, camber_loc_input.value(), thickness_input.value()))

        control_layout.addWidget(camber_label)
        control_layout.addWidget(camber_input)

        camber_loc_label = QLabel("Camber Location (tenths of chord):")
        camber_loc_input = QDoubleSpinBox()
        camber_loc_input.setRange(1, 9)  # p = 0 or 1 makes the camber equations singular
        camber_loc_input.setSingleStep(0.5)
        camber_loc_input.setValue(5)
        camber_loc_input.valueChanged.connect(lambda value: self.update_airfoil_plot(self.airfoil_plot, camber_input.value(), value, thickness_input.value()))
        control_layout.addWidget(camber_loc_label)
        control_layout.addWidget(camber_loc_input)

        thickness_label = QLabel("Thickness:")
        thickness_input = QDoubleSpinBox()
        thickness_input.setRange(0.01, 0.30)
        thickness_input.setValue(0.12)
        thickness_input.setSingleStep(0.01)
        thickness_input.valueChanged.connect(lambda value: self.update_airfoil_plot(self.airfoil_plot, camber_input.value(), camber_loc_input.value(), value))
        control_layout.addWidget(thickness_label)
        control_layout.addWidget(thickness_input)

        self.inducing_input = QSpinBox()
        self.inducing_input.setRange(2, 100)
        self.inducing_input.setValue(8)
        control_layout.addWidget(QLabel("Inducing points (taps) per surface:"))
        control_layout.addWidget(self.inducing_input)

        self.steps_input = QSpinBox()
        self.steps_input.setRange(50, 5000)
        self.steps_input.setSingleStep(50)
        self.steps_input.setValue(400)
        control_layout.addWidget(QLabel("Training steps:"))
        control_layout.addWidget(self.steps_input)

        self.train_button = QPushButton("Train GP")
        self.train_button.clicked.connect(self._on_train_clicked)
        control_layout.addWidget(self.train_button)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)

        control_widget = QWidget()
        control_widget.setLayout(control_layout)

        side_layout.addWidget(control_widget)

        self.setCentralWidget(main_widget)

    def update_airfoil_plot(self, airfoil_plot, camber, camber_loc, thickness):
        # NACA digits: camber in % of chord, its location in tenths of chord, thickness as a fraction
        airfoil = naca4(camber / 100.0, camber_loc / 10.0, thickness)

        # Placeholder pressure from thin airfoil theory + a thickness term (not CFD)
        upper_press, lower_press = placeholder_cp(airfoil)

        # A new airfoil invalidates any trained GP and its taps
        self._cancel_training()
        self._surface = surface_data(airfoil, upper_press, lower_press)
        self.status_label.setText("")
        self.training_plot.reset(self.steps_input.value())

        airfoil_plot.set_airfoil(airfoil)
        airfoil_plot.clear_taps()
        self.pressure_plot.set_pressure(airfoil.x, upper_press, lower_press)
        self.pressure_plot.clear_gp()

    # --- training ---------------------------------------------------------------------

    def _on_train_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()  # the run reports back through training_done
            return
        self._start_training()

    def _start_training(self) -> None:
        surface = self._surface
        if surface is None:
            return
        self._cancel_training()
        self.pressure_plot.clear_gp()
        self.airfoil_plot.clear_taps()

        steps = self.steps_input.value()
        self._grids = (
            np.linspace(0.0, surface.upper.s[-1], GRID_POINTS),
            np.linspace(0.0, surface.lower.s[-1], GRID_POINTS),
        )
        self.training_plot.reset(steps)
        self._last_progress = None

        self._run_id += 1
        worker = TrainWorker(
            self._run_id, surface, *self._grids, self.inducing_input.value(), steps
        )
        worker.progress.connect(self._on_progress)
        worker.training_done.connect(self._on_training_done)
        worker.finished.connect(lambda w=worker: self._on_thread_finished(w))
        self._worker = worker
        self._set_training_ui(True)
        worker.start()

    def _cancel_training(self) -> None:
        """Stop any running training and make the UI ignore whatever it still has queued."""
        self._run_id += 1
        worker = self._worker
        if worker is not None:
            worker.stop()
            worker.wait()
            self._worker = None
        self._set_training_ui(False)

    def _set_training_ui(self, running: bool) -> None:
        self.train_button.setText("Stop" if running else "Train GP")
        self.inducing_input.setEnabled(not running)
        self.steps_input.setEnabled(not running)

    def _on_progress(self, progress: Progress) -> None:
        if progress.run_id != self._run_id:
            return
        self._last_progress = progress
        self.training_plot.append(progress.step, progress.upper.elbo, progress.lower.elbo)
        self._show_gp(progress, final=False)
        self.status_label.setText(
            f"Step {progress.step + 1}/{progress.steps}   "
            f"ELBO/pt upper {progress.upper.elbo:.3f}, lower {progress.lower.elbo:.3f}"
        )

    def _on_training_done(self, run_id: int) -> None:
        if run_id != self._run_id:
            return
        self._set_training_ui(False)
        progress = self._last_progress
        if progress is None or self._surface is None:
            return
        self._show_gp(progress, final=True)

        def tap_positions(side_progress: SurfaceProgress, surface: Surface) -> str:
            x = np.sort(np.interp(side_progress.tap_s, surface.s, surface.x))
            return ", ".join(f"{value:.2f}" for value in x)

        upper, lower = progress.upper, progress.lower
        self.status_label.setText(
            f"Done: {len(upper.tap_s)} taps per surface\n"
            f"Upper (ELBO/pt {upper.elbo:.3f}) x/c: {tap_positions(upper, self._surface.upper)}\n"
            f"Lower (ELBO/pt {lower.elbo:.3f}) x/c: {tap_positions(lower, self._surface.lower)}"
        )

    def _on_thread_finished(self, worker: TrainWorker) -> None:
        if self._worker is worker:
            self._worker = None

    def _show_gp(self, progress: Progress, final: bool) -> None:
        surface = self._surface
        if surface is None:
            return
        sides = (
            (surface.upper, progress.upper, self._grids[0]),
            (surface.lower, progress.lower, self._grids[1]),
        )
        self.pressure_plot.set_gp(
            *(
                (np.interp(grid, s.s, s.x), p.grid_mean, p.grid_std, p.grid_linear)
                for s, p, grid in sides
            )
        )

        tap_x = np.concatenate([np.interp(p.tap_s, s.s, s.x) for s, p, _ in sides])
        tap_y = np.concatenate([np.interp(p.tap_s, s.s, s.y) for s, p, _ in sides])
        tap_cp = np.concatenate([p.tap_cp for _, p, _ in sides])
        self.pressure_plot.set_taps(tap_x, tap_cp, final)
        self.airfoil_plot.set_taps(tap_x, tap_y, final)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._cancel_training()
        super().closeEvent(event)


class FixedPlot(PlotWidget):
    """Plot with a fixed view (no auto-range, pan or zoom), coloured from the window palette."""

    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        x_label: str,
        y_label: str,
    ) -> None:
        super().__init__()
        plot_item = self.getPlotItem()
        if not plot_item:
            raise RuntimeError("Failed to get PlotItem from PlotWidget")

        # Match the window: take background and text colours from the palette
        palette = self.palette()
        self.background = palette.color(QPalette.ColorRole.Window)
        self.foreground = palette.color(QPalette.ColorRole.WindowText)
        self.setBackground(self.background)

        plot_item.disableAutoRange()
        plot_item.setXRange(*x_range, padding=0)
        plot_item.setYRange(*y_range, padding=0)
        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.setMenuEnabled(False)
        plot_item.hideButtons()

        plot_item.showGrid(x=True, y=True, alpha=0.2)
        plot_item.setLabel("bottom", x_label)
        plot_item.setLabel("left", y_label)
        for axis in ("left", "bottom"):
            plot_item.getAxis(axis).setPen(pg.mkPen(with_alpha(self.foreground, 140)))
            plot_item.getAxis(axis).setTextPen(pg.mkPen(self.foreground))
        plot_item.getAxis("left").setWidth(60)  # same width so the plot areas line up


class AirfoilPlot(FixedPlot):
    def __init__(self) -> None:
        super().__init__(X_RANGE, AIRFOIL_Y_RANGE, "x / c", "y / c")
        pen = pg.mkPen(self.foreground, width=2)
        self.upper = self.plot(pen=pen)
        self.lower = self.plot(pen=pen)
        self.addItem(
            pg.FillBetweenItem(self.upper, self.lower, brush=pg.mkBrush(with_alpha(self.foreground, 30)))
        )
        self.taps = pg.ScatterPlotItem()
        self.taps.setZValue(10)
        self.addItem(self.taps)
        self._tap_xy: tuple[Array, Array] | None = None

    def set_airfoil(self, airfoil: Airfoil) -> None:
        self.upper.setData(airfoil.x_upper, airfoil.y_upper)
        self.lower.setData(airfoil.x_lower, airfoil.y_lower)

    def set_taps(self, x: Array, y: Array, final: bool) -> None:
        self._tap_xy = (x, y)
        self.taps.setData(x=x, y=y, **tap_style(final, self.foreground))

    def clear_taps(self) -> None:
        self._tap_xy = None
        self.taps.setData(x=np.empty(0), y=np.empty(0))


class PressurePlot(FixedPlot):
    def __init__(self) -> None:
        super().__init__(X_RANGE, PRESSURE_Y_RANGE, "x / c", "Cp")
        legend = self.getPlotItem().addLegend(offset=(-10, 10))
        legend.setBrush(pg.mkBrush(with_alpha(self.background, 200)))
        legend.setPen(pg.mkPen(with_alpha(self.foreground, 90)))

        # Pressure data
        self.upper = self.plot(pen=pg.mkPen(BLUE, width=2), name="Upper pressure")
        self.lower = self.plot(pen=pg.mkPen(RED, width=2), name="Lower pressure")
        self.addItem(pg.FillBetweenItem(self.upper, self.lower, brush=(94, 155, 255, 45)))

        # LMGP fit, per surface: the linear model alone (dotted), then linear model + GP
        # as a dashed mean with a +-2 sigma band
        dotted = pg.mkPen(self.foreground, width=1, style=Qt.PenStyle.DotLine)
        self.linear_upper = self.plot(pen=dotted, name="Linear model")
        self.linear_lower = self.plot(pen=dotted)
        dashed = pg.mkPen(self.foreground, width=2, style=Qt.PenStyle.DashLine)
        self.mean_upper = self.plot(pen=dashed, name="LMGP mean")
        self.mean_lower = self.plot(pen=dashed)
        self.band_upper = self._make_band()
        self.band_lower = self._make_band()

        self.taps = pg.ScatterPlotItem(name="Inducing points")
        self.taps.setZValue(10)
        self.addItem(self.taps)
        self._tap_xy: tuple[Array, Array] | None = None

    def _make_band(self) -> tuple[pg.PlotDataItem, pg.PlotDataItem]:
        low = self.plot(pen=None)
        high = self.plot(pen=None)
        self.addItem(pg.FillBetweenItem(low, high, brush=pg.mkBrush(with_alpha(self.foreground, 45))))
        return low, high

    def set_pressure(self, x: Array, upper_press: Array, lower_press: Array) -> None:
        self.upper.setData(x, upper_press)
        self.lower.setData(x, lower_press)

    def set_gp(
        self,
        upper: tuple[Array, Array, Array, Array],
        lower: tuple[Array, Array, Array, Array],
    ) -> None:
        """Each side is (x, mean, std, linear): the full mean, GP std and the linear part."""
        for (x, mean, std, linear), mean_curve, linear_curve, (low, high) in (
            (upper, self.mean_upper, self.linear_upper, self.band_upper),
            (lower, self.mean_lower, self.linear_lower, self.band_lower),
        ):
            linear_curve.setData(x, linear)
            mean_curve.setData(x, mean)
            low.setData(x, mean - 2.0 * std)
            high.setData(x, mean + 2.0 * std)

    def set_taps(self, x: Array, cp: Array, final: bool) -> None:
        self._tap_xy = (x, cp)
        self.taps.setData(x=x, y=cp, **tap_style(final, self.foreground))

    def clear_gp(self) -> None:
        empty = np.empty(0)
        for curve in (
            self.linear_upper,
            self.linear_lower,
            self.mean_upper,
            self.mean_lower,
            *self.band_upper,
            *self.band_lower,
        ):
            curve.setData(empty, empty)
        self._tap_xy = None
        self.taps.setData(x=empty, y=empty)


class TrainingPlot(FixedPlot):
    """Live ELBO curves, one per surface. The x-range is the step budget; y follows recent values."""

    def __init__(self) -> None:
        super().__init__((0, 1), (-1, 1), "Step", "ELBO / point")
        legend = self.getPlotItem().addLegend(offset=(-10, 10))
        legend.setBrush(pg.mkBrush(with_alpha(self.background, 200)))
        legend.setPen(pg.mkPen(with_alpha(self.foreground, 90)))
        self.curve_upper = self.plot(pen=pg.mkPen(BLUE, width=2), name="Upper")
        self.curve_lower = self.plot(pen=pg.mkPen(RED, width=2), name="Lower")
        self._steps: list[float] = []
        self._elbo_upper: list[float] = []
        self._elbo_lower: list[float] = []

    def reset(self, total_steps: int) -> None:
        self._steps, self._elbo_upper, self._elbo_lower = [], [], []
        self.curve_upper.setData([], [])
        self.curve_lower.setData([], [])
        plot_item = self.getPlotItem()
        plot_item.setXRange(0, total_steps, padding=0)
        plot_item.setYRange(-1, 1, padding=0)

    def append(self, step: int, elbo_upper: float, elbo_lower: float) -> None:
        self._steps.append(step)
        self._elbo_upper.append(elbo_upper)
        self._elbo_lower.append(elbo_lower)
        self.curve_upper.setData(self._steps, self._elbo_upper)
        self.curve_lower.setData(self._steps, self._elbo_lower)

        # Skip the steep start so the later improvement stays readable
        skip = len(self._steps) // 5
        recent = self._elbo_upper[skip:] + self._elbo_lower[skip:]
        low, high = min(recent), max(recent)
        pad = 0.1 * max(high - low, 1e-3)
        self.getPlotItem().setYRange(low - pad, high + pad, padding=0)
