"""Load a saved run and replay its policy, optionally against the uncontrolled baseline."""

from pathlib import Path

import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from torch import Tensor

from src.ml.envs.base import Trace
from src.ml.evaluate import EvalResult, evaluate
from src.ml.runs import list_runs, load_run, make_env
from src.ui.plotting import plot_batch, series_pen

POLICY_PEN = 0
BASELINE_COLOUR = (150, 150, 150)


def _channels(trace: Trace) -> dict[str, Tensor]:
    """Every signal to plot, each [T, N]. Extras named '<x>_ref' are drawn as targets on '<x>', not on their own."""
    out = {k: v for k, v in trace.extras.items() if not (k.endswith("_ref") and k[:-4] in trace.extras)}
    for j in range(trace.action.shape[-1]):
        out[f"action[{j}]" if trace.action.shape[-1] > 1 else "action"] = trace.action[..., j]
    out["reward"] = trace.reward
    return out


class RolloutTab(QWidget):
    def __init__(self) -> None:
        super().__init__()

        # --- controls ---
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(280)
        refresh = QPushButton("↻")
        refresh.setFixedWidth(30)
        refresh.setToolTip("Refresh run list")
        refresh.clicked.connect(self.refresh_runs)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_combo, 1)
        run_row.addWidget(refresh)

        self.n_envs = QSpinBox()
        self.n_envs.setRange(1, 1024)
        self.n_envs.setValue(8)
        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        self.seed.setValue(999)
        self.deterministic = QCheckBox("Deterministic (mean action)")
        self.deterministic.setChecked(True)
        self.baseline = QCheckBox("Compare to zero-action baseline")
        self.baseline.setChecked(True)
        run_btn = QPushButton("Run rollout")
        run_btn.clicked.connect(self.run_rollout)

        form = QFormLayout()
        form.addRow("Run", run_row)
        form.addRow("Parallel episodes", self.n_envs)
        form.addRow("Seed", self.seed)
        form.addRow(self.deterministic)
        form.addRow(self.baseline)
        form.addRow(run_btn)

        self.results = QTableWidget(0, 3)
        self.results.setHorizontalHeaderLabels(["Metric", "Policy", "Baseline"])
        self.results.verticalHeader().setVisible(False)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addLayout(form)
        controls_layout.addWidget(QLabel("Results"))
        controls_layout.addWidget(self.results, 1)

        # --- plots ---
        self.plots = pg.GraphicsLayoutWidget()

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(controls)
        split.addWidget(self.plots)
        split.setSizes([330, 870])
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        self.refresh_runs()

    # ----- public -----
    def refresh_runs(self) -> None:
        current = self.run_combo.currentData()
        self.run_combo.clear()
        for s in list_runs():
            if s.run.policy_path.exists():
                self.run_combo.addItem(f"{s.run.name}  [{s.status.state}]", s.run.path)
        if current is not None:
            self.select_run(current)

    def select_run(self, path: Path) -> None:
        idx = self.run_combo.findData(path)
        if idx >= 0:
            self.run_combo.setCurrentIndex(idx)

    def run_rollout(self) -> None:
        path: Path | None = self.run_combo.currentData()
        if path is None:
            return
        try:
            loaded = load_run(path)
        except Exception as e:  # show any load problem instead of crashing the UI
            QMessageBox.warning(self, "Could not load run", repr(e))
            return

        env = make_env(loaded.config, n_envs=self.n_envs.value())
        seed = self.seed.value()
        result, trace = evaluate(env, loaded.policy.as_act_fn(self.deterministic.isChecked()), seed)
        base: tuple[EvalResult, Trace] | None = evaluate(env, None, seed) if self.baseline.isChecked() else None
        self._show_results(result, base[0] if base else None)
        self._plot(trace, base[1] if base else None)

    # ----- internals -----
    def _show_results(self, res: EvalResult, base: EvalResult | None) -> None:
        def rows(r: EvalResult) -> dict[str, float]:
            return {"mean return": r.mean_return, "action rms": r.action_rms, **r.env_metrics}

        policy_rows = rows(res)
        base_rows = rows(base) if base else {}
        self.results.setRowCount(len(policy_rows))
        for i, (k, v) in enumerate(policy_rows.items()):
            b = base_rows.get(k)
            for col, text in enumerate([k, f"{v:.4g}", "" if b is None else f"{b:.4g}"]):
                self.results.setItem(i, col, QTableWidgetItem(text))
        self.results.resizeColumnsToContents()

    def _plot(self, trace: Trace, base: Trace | None) -> None:
        self.plots.clear()
        channels = _channels(trace)
        base_channels = _channels(base) if base else {}
        first: pg.PlotItem | None = None
        for row, (name, y) in enumerate(channels.items()):
            p: pg.PlotItem = self.plots.addPlot(row=row, col=0)
            p.setLabel("left", name)
            p.showGrid(x=True, y=True, alpha=0.3)
            if row == 0:
                p.addLegend(offset=(-10, 10))
            if first is None:
                first = p
            else:
                p.setXLink(first)
            if name in base_channels:
                plot_batch(p, trace.t, base_channels[name], pg.mkPen(BASELINE_COLOUR, width=1), "baseline" if row == 0 else None)
            plot_batch(p, trace.t, y, series_pen(POLICY_PEN), "policy" if row == 0 else None)
            ref = trace.extras.get(f"{name}_ref")
            if ref is not None:
                plot_batch(p, trace.t, ref[:, :1], pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine), "target" if row == 0 else None)
        if first is not None:
            last: pg.PlotItem = self.plots.getItem(len(channels) - 1, 0)
            last.setLabel("bottom", "time [s]")
