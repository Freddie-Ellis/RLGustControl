"""Compare any number of saved runs: their full training histories on one shared x-axis (transitions), a summary per
run, and the config settings that differ. Every logged value is drawn as it is, nothing smoothed or left out.
All the comparing is done by src/ml/compare.py; this tab only picks runs and draws the results."""

from pathlib import Path

import polars as pl
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.ml import compare
from src.ml.runs import RunSummary, list_runs
from src.ui.plotting import series_pen
from src.ui.utils.dynamic_vb import DynamicViewBox

LIVE_MS = 2000  # check this often for new rows while a compared run is still running
POINT_SIZE = 4  # every logged value is also drawn as a point, so a lone eval episode is not lost in the line
X_TOOLTIP = (
    "Data collected so far: one transition per control step (obs, action, reward, next obs), over all parallel envs.\n"
    "The same unit for every kind of run, so it compares how much data each needed."
)


def _fill_table(table: QTableWidget, df: pl.DataFrame) -> None:
    table.clear()
    table.setRowCount(df.height)
    table.setColumnCount(df.width)
    table.setHorizontalHeaderLabels(df.columns)
    for r, row in enumerate(df.iter_rows()):
        for c, v in enumerate(row):
            text = "" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
            table.setItem(r, c, QTableWidgetItem(text))
    table.resizeColumnsToContents()


def _signature(path: Path) -> tuple[float, int] | None:
    """Changes whenever the file does (an appended metrics row): its modification time and size."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime, st.st_size


class CompareTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._runs: list[RunSummary] = []
        self._curves: list[tuple[str, RunSummary, pg.PlotDataItem]] = []  # metric, run, its curve
        self._signatures: dict[Path, tuple[float, int] | None] = {}  # metrics file state the curves were drawn from

        # --- left: runs and metrics ---
        self.run_list = QListWidget()
        self.run_list.itemChanged.connect(lambda _: self._on_runs_changed())
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self._set_all_checked(False))
        run_buttons = QHBoxLayout()
        run_buttons.addWidget(refresh)
        run_buttons.addWidget(clear)
        run_buttons.addStretch()

        self.metric_list = QListWidget()
        self.metric_list.itemChanged.connect(lambda _: self._plot())

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Runs (tick to compare)"))
        left_layout.addLayout(run_buttons)
        left_layout.addWidget(self.run_list, 3)
        left_layout.addWidget(QLabel("Metrics"))
        left_layout.addWidget(self.metric_list, 2)

        # --- right: history panels over summary / config differences ---
        self.plots = pg.GraphicsLayoutWidget()
        self.note = QLabel("")
        self.note.setStyleSheet("color: #e57373")
        history = QWidget()
        history_layout = QVBoxLayout(history)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.addWidget(self.plots, 1)
        history_layout.addWidget(self.note)

        self.summary = QTableWidget()
        self.differences = QTableWidget()
        for table in (self.summary, self.differences):
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.verticalHeader().setVisible(False)
        self.differences.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        tables = QTabWidget()
        tables.addTab(self.summary, "Summary")
        tables.addTab(self.differences, "Config differences")

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(history)
        right.addWidget(tables)
        right.setSizes([560, 240])

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([300, 1000])
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(LIVE_MS)
        self.refresh()

    # ----- runs -----
    def refresh(self) -> None:
        """Re-read the runs folder, keeping what was ticked."""
        ticked = {s.run.path for s in self._checked()}
        self._runs = list_runs()
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for s in self._runs:
            item = QListWidgetItem(f"{s.run.name}  [{s.status.state}]  {s.config.describe()}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if s.run.path in ticked else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, s.run.path)
            self.run_list.addItem(item)
        self.run_list.blockSignals(False)
        self._on_runs_changed()

    def _checked(self) -> list[RunSummary]:
        """The ticked runs, in list order (which also fixes each run's colour)."""
        paths: set[Path] = set()
        for i in range(self.run_list.count()):
            item = self.run_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                paths.add(item.data(Qt.ItemDataRole.UserRole))
        return [s for s in self._runs if s.run.path in paths]

    def _set_all_checked(self, checked: bool) -> None:
        self.run_list.blockSignals(True)
        for i in range(self.run_list.count()):
            self.run_list.item(i).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.run_list.blockSignals(False)
        self._on_runs_changed()

    def _on_runs_changed(self) -> None:
        runs = self._checked()
        self._fill_metrics(compare.metric_names(runs))
        _fill_table(self.summary, compare.summary(runs))
        differences = compare.config_differences(runs) if len(runs) > 1 else pl.DataFrame()
        if len(runs) > 1 and differences.is_empty():
            differences = pl.DataFrame({"setting": ["(every setting is the same)"]})
        _fill_table(self.differences, differences)
        self._plot()

    def _fill_metrics(self, names: list[str]) -> None:
        """Offer every metric the ticked runs logged. Keep the ticks; tick the defaults the first time."""
        current = {self.metric_list.item(i).text(): self.metric_list.item(i).checkState()
                   for i in range(self.metric_list.count())}
        self.metric_list.blockSignals(True)
        self.metric_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            default = Qt.CheckState.Checked if name in compare.DEFAULT_METRICS else Qt.CheckState.Unchecked
            item.setCheckState(current.get(name, default))
            self.metric_list.addItem(item)
        self.metric_list.blockSignals(False)

    def _metrics(self) -> list[str]:
        return [self.metric_list.item(i).text() for i in range(self.metric_list.count())
                if self.metric_list.item(i).checkState() == Qt.CheckState.Checked]

    # ----- plots -----
    def _plot(self) -> None:
        """Build one panel per ticked metric, every ticked run overlaid in its own colour, x-axes linked. Only when the
        ticked runs or metrics change; new rows while a run trains go in through `_update_curves`, so a zoom is
        kept."""
        self.plots.clear()
        self._curves = []
        runs, metrics = self._checked(), self._metrics()
        first: pg.PlotItem | None = None
        for row, metric in enumerate(metrics):
            p: pg.PlotItem = self.plots.addPlot(row=row, col=0, viewBox=DynamicViewBox())
            p.setLabel("left", metric)
            p.showGrid(x=True, y=True, alpha=0.3)
            if first is None:
                first = p
                p.addLegend(offset=(-10, 10))
            else:
                p.setXLink(first)
            for i, s in enumerate(runs):  # an item per run even before it has data, so live rows can fill it
                pen = series_pen(i)
                line = p.plot([], [], pen=pen, symbol="o", symbolSize=POINT_SIZE, symbolPen=None,
                              symbolBrush=pen.color(), name=s.run.name if row == 0 else None)
                self._curves.append((metric, s, line))
        if metrics:
            last: pg.PlotItem = self.plots.getItem(len(metrics) - 1, 0)
            last.setLabel("bottom", compare.X_LABEL)
            last.getAxis("bottom").setToolTip(X_TOOLTIP)
        self._update_curves()

    def _update_curves(self) -> None:
        """(Re)fill the existing curves with every row the runs have logged. Views still auto-ranging follow the
        data; one the user has zoomed or panned keeps its range (double-click fits it again)."""
        runs = {s.run.path: s for _, s, _ in self._curves}
        histories = {path: s.run.read_metrics() for path, s in runs.items()}
        self._signatures = {path: _signature(s.run.metrics_path) for path, s in runs.items()}
        no_axis = [s.run.name for path, s in runs.items() if compare.x_column(histories[path]) is None]
        self.note.setText(f"Logged no transitions, so cannot be placed: {', '.join(no_axis)}" if no_axis else "")
        for metric, s, line in self._curves:
            c = compare.curve(histories[s.run.path], metric, s.config.algo.updates_per_step)
            line.setData(c.x if c else [], c.y if c else [])

    def _tick(self) -> None:
        """While a ticked run is still running, take in its new rows, but only when its metrics file has changed (a
        run whose learner died without saying so stays "running" forever, and must not redraw anything)."""
        runs = self._checked()
        if not self.isVisible() or not any(s.status.state == "running" for s in runs):
            return
        for s in runs:
            try:
                s.status = s.run.read_status()
            except (OSError, ValueError):  # caught mid-write: keep the last one
                pass
        if all(_signature(s.run.metrics_path) == self._signatures.get(s.run.path) for s in runs):
            return
        _fill_table(self.summary, compare.summary(runs))
        self._update_curves()
