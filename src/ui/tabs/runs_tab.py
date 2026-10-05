"""Browse saved runs, overlay their training curves and read their configs."""

from pathlib import Path

import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.ml.runs import RunSummary, list_runs
from src.ui.plotting import series_pen

COLUMNS = ["Run", "Status", "Env", "Policy", "Algo", "Iters", "Final return"]


class RunsTab(QWidget):
    replay_requested = Signal(Path)  # run directory
    template_requested = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self._runs: list[RunSummary] = []

        # --- left: run table + buttons ---
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_selection)

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        self.replay_btn = QPushButton("Replay")
        self.replay_btn.clicked.connect(self._replay)
        self.template_btn = QPushButton("Use as template")
        self.template_btn.clicked.connect(self._use_as_template)

        buttons = QHBoxLayout()
        for b in (refresh, self.replay_btn, self.template_btn):
            buttons.addWidget(b)
        buttons.addStretch()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.table)

        # --- right: training curves + config ---
        self.metric = QComboBox()
        self.metric.currentTextChanged.connect(lambda _: self._plot())
        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.setLabel("bottom", "iteration")
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        self.config_view = QPlainTextEdit()
        self.config_view.setReadOnly(True)
        self.config_view.setFont(QFont("Consolas", 9))

        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("Metric"))
        metric_row.addWidget(self.metric, 1)

        right_split = QSplitter(Qt.Orientation.Vertical)
        curves = QWidget()
        curves_layout = QVBoxLayout(curves)
        curves_layout.addLayout(metric_row)
        curves_layout.addWidget(self.plot)
        right_split.addWidget(curves)
        right_split.addWidget(self.config_view)
        right_split.setSizes([500, 250])

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right_split)
        split.setSizes([550, 650])
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        # Re-plot periodically while a selected run is still training.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(2000)

        self.refresh()

    # ----- public -----
    def refresh(self) -> None:
        selected = {s.run.path for s in self._selected()}
        self._runs = list_runs()
        self.table.setRowCount(len(self._runs))
        for row, s in enumerate(self._runs):
            final = "" if s.status.final_return is None else f"{s.status.final_return:.3f}"
            values = [
                s.run.name, s.status.state, s.config.env.kind, s.config.policy.kind,
                s.config.algo.kind, str(s.status.iterations_done), final,
            ]
            for col, v in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(v))
            if s.run.path in selected:
                self.table.selectRow(row)
        self.table.resizeColumnsToContents()
        self._on_selection()

    # ----- internals -----
    def _selected(self) -> list[RunSummary]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self._runs[r] for r in rows if r < len(self._runs)]

    def _replay(self) -> None:
        if sel := self._selected():
            self.replay_requested.emit(sel[0].run.path)

    def _use_as_template(self) -> None:
        if sel := self._selected():
            self.template_requested.emit(sel[0].run.path)

    def _on_selection(self) -> None:
        sel = self._selected()
        self.replay_btn.setEnabled(len(sel) == 1 and sel[0].run.policy_path.exists())
        self.template_btn.setEnabled(len(sel) == 1)
        self.config_view.setPlainText(sel[0].run.config_path.read_text() if len(sel) == 1 else "")

        # Offer the metric columns common to all selected runs.
        columns: set[str] | None = None
        for s in sel:
            cols = set(s.run.read_metrics().columns) - {"iteration"}
            columns = cols if columns is None else columns & cols
        current = self.metric.currentText()
        self.metric.blockSignals(True)
        self.metric.clear()
        self.metric.addItems(sorted(columns or []))
        if current in (columns or set()):
            self.metric.setCurrentText(current)
        elif "train_return" in (columns or set()):
            self.metric.setCurrentText("train_return")
        self.metric.blockSignals(False)
        self._plot()

    def _plot(self) -> None:
        self.plot.clear()
        metric = self.metric.currentText()
        if not metric:
            return
        self.plot.setLabel("left", metric)
        for i, s in enumerate(self._selected()):
            df = s.run.read_metrics()
            if metric in df.columns:
                self.plot.plot(df["iteration"].to_numpy(), df[metric].to_numpy(), pen=series_pen(i), name=s.run.name)

    def _tick(self) -> None:
        if self.isVisible() and any(s.status.state == "running" for s in self._selected()):
            self._plot()
