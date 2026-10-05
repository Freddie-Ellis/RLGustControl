"""Open a sensor / CFD / experiment table, preview it, plot columns and run integrity checks."""

from pathlib import Path

import numpy as np
import polars as pl
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.config import ROOT
from src.data.checks import Level, check_table
from src.data.loaders import SUPPORTED_SUFFIXES, load_table
from src.ui.plotting import series_pen

PREVIEW_ROWS = 200
NO_TIME = "(row index)"
LEVEL_COLOURS: dict[Level, str] = {"ok": "#2e7d32", "info": "#555555", "warning": "#ef6c00", "error": "#c62828"}


class DataTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._df: pl.DataFrame | None = None

        open_btn = QPushButton("Open file…")
        open_btn.clicked.connect(self._open)
        self.file_label = QLabel("No file loaded")
        self.time_combo = QComboBox()
        self.time_combo.currentTextChanged.connect(lambda _: self._refresh())
        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(self.file_label, 1)
        top.addWidget(QLabel("Time column"))
        top.addWidget(self.time_combo)

        self.columns = QListWidget()
        self.columns.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.columns.itemSelectionChanged.connect(self._plot)
        self.checks = QListWidget()
        left_split = QSplitter(Qt.Orientation.Vertical)
        for title, widget in (("Columns (select to plot)", self.columns), ("Checks", self.checks)):
            box = QWidget()
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(0, 0, 0, 0)
            box_layout.addWidget(QLabel(title))
            box_layout.addWidget(widget)
            left_split.addWidget(box)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.preview = QTableWidget()
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(self.plot)
        right_split.addWidget(self.preview)
        right_split.setSizes([500, 250])

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_split)
        split.addWidget(right_split)
        split.setSizes([320, 880])

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split, 1)

    def load(self, path: Path) -> None:
        try:
            df = load_table(path)
        except Exception as e:  # show any read problem instead of crashing the UI
            QMessageBox.warning(self, "Could not load file", repr(e))
            return
        self._df = df
        self.file_label.setText(f"{path.name}  ({df.height} rows x {df.width} columns)")

        numeric = [c for c, dt in df.schema.items() if dt.is_numeric()]
        self.time_combo.blockSignals(True)
        self.time_combo.clear()
        self.time_combo.addItems([NO_TIME, *numeric])
        guess = next((c for c in numeric if c.lower() in ("t", "time", "time_s")), NO_TIME)
        self.time_combo.setCurrentText(guess)
        self.time_combo.blockSignals(False)

        self._fill_preview(df)
        self._refresh()

    # ----- internals -----
    def _open(self) -> None:
        patterns = " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES)
        name, _ = QFileDialog.getOpenFileName(self, "Open dataset", str(ROOT), f"Tables ({patterns})")
        if name:
            self.load(Path(name))

    def _time_col(self) -> str | None:
        t = self.time_combo.currentText()
        return None if t in ("", NO_TIME) else t

    def _refresh(self) -> None:
        """Re-run checks and rebuild the column list after loading or changing the time column."""
        if self._df is None:
            return
        time_col = self._time_col()
        self.checks.clear()
        for f in check_table(self._df, time_col):
            item = QListWidgetItem(f"[{f.level}] {f.message}")
            item.setForeground(QColor(LEVEL_COLOURS[f.level]))
            self.checks.addItem(item)

        self.columns.clear()
        for c, dt in self._df.schema.items():
            if dt.is_numeric() and c != time_col:
                self.columns.addItem(c)
        self._plot()

    def _plot(self) -> None:
        self.plot.clear()
        if self._df is None:
            return
        time_col = self._time_col()
        x = self._df[time_col].to_numpy() if time_col else np.arange(self._df.height)
        self.plot.setLabel("bottom", time_col or "row")
        for i, item in enumerate(self.columns.selectedItems()):
            col = item.text()
            y = self._df[col].cast(pl.Float64).to_numpy()
            self.plot.plot(x, y, pen=series_pen(i), name=col, connect="finite")

    def _fill_preview(self, df: pl.DataFrame) -> None:
        head = df.head(PREVIEW_ROWS)
        self.preview.clear()
        self.preview.setRowCount(head.height)
        self.preview.setColumnCount(head.width)
        self.preview.setHorizontalHeaderLabels(head.columns)
        for r, row in enumerate(head.iter_rows()):
            for c, v in enumerate(row):
                self.preview.setItem(r, c, QTableWidgetItem("" if v is None else str(v)))
        self.preview.resizeColumnsToContents()
