"""Edit a run config, validate it, and launch training in a separate process with a live curve."""

import sys
import tempfile
import tomllib
from pathlib import Path

import pydantic
import pyqtgraph as pg
from qtpy.QtCore import QProcess, Qt, QTimer, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.config import CONFIG, ROOT
from src.ml.run_config import RunConfig
from src.ml.runs import RunDir
from src.ui.plotting import series_pen

MONO = QFont("Consolas", 10)


class TrainTab(QWidget):
    run_started = Signal(Path)
    run_finished = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self._process: QProcess | None = None
        self._run: RunDir | None = None
        self._stdout_buffer = ""

        # --- config editor ---
        self.template_combo = QComboBox()
        self.template_combo.currentIndexChanged.connect(self._load_template)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.clicked.connect(self.refresh_templates)
        template_row = QHBoxLayout()
        template_row.addWidget(QLabel("Config"))
        template_row.addWidget(self.template_combo, 1)
        template_row.addWidget(refresh_btn)

        self.editor = QPlainTextEdit()
        self.editor.setFont(MONO)
        self.editor.textChanged.connect(lambda: self.validation.setText(""))

        self.validation = QLabel()
        self.validation.setWordWrap(True)
        self.validation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        validate_btn = QPushButton("Validate")
        validate_btn.clicked.connect(self.validate)
        save_btn = QPushButton("Save as…")
        save_btn.clicked.connect(self._save_as)
        self.launch_btn = QPushButton("Launch training")
        self.launch_btn.clicked.connect(self.launch)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        action_row = QHBoxLayout()
        for b in (validate_btn, save_btn):
            action_row.addWidget(b)
        action_row.addStretch()
        action_row.addWidget(self.launch_btn)
        action_row.addWidget(self.stop_btn)

        editor_panel = QWidget()
        editor_layout = QVBoxLayout(editor_panel)
        editor_layout.addLayout(template_row)
        editor_layout.addWidget(self.editor, 1)
        editor_layout.addWidget(self.validation)
        editor_layout.addLayout(action_row)

        # --- live progress ---
        self.status = QLabel("Idle")
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "iteration")
        self.plot.setLabel("left", "train_return")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=series_pen(0, width=2))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMaximumBlockCount(5000)

        progress_split = QSplitter(Qt.Orientation.Vertical)
        progress_split.addWidget(self.plot)
        progress_split.addWidget(self.log)
        progress_panel = QWidget()
        progress_layout = QVBoxLayout(progress_panel)
        progress_layout.addWidget(self.status)
        progress_layout.addWidget(progress_split, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(editor_panel)
        split.addWidget(progress_panel)
        split.setSizes([500, 700])
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_curve)

        self.refresh_templates()
        self._load_template()

    # ----- public -----
    def refresh_templates(self) -> None:
        current = self.template_combo.currentText()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        for p in sorted(CONFIG.configs_dir.glob("*.toml")):
            self.template_combo.addItem(p.name, p)
        if current:
            self.template_combo.setCurrentText(current)
        self.template_combo.blockSignals(False)

    def load_from_run(self, run_path: Path) -> None:
        """Fill the editor with an existing run's config (from the Runs tab's 'Use as template')."""
        cfg = RunDir(run_path).read_config()
        self.editor.setPlainText(cfg.to_toml())
        self.validation.setText(f"Loaded config of run {run_path.name}")

    def validate(self) -> RunConfig | None:
        try:
            cfg = RunConfig.from_toml_str(self.editor.toPlainText())
        except tomllib.TOMLDecodeError as e:
            self._set_validation(False, f"TOML syntax error: {e}")
            return None
        except pydantic.ValidationError as e:
            msgs = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
            self._set_validation(False, "\n".join(msgs))
            return None
        self._set_validation(True, f"Valid: run '{cfg.name}', {cfg.algo.kind} for {cfg.algo.iterations} iterations")
        return cfg

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    def launch(self) -> None:
        if self.is_running() or self.validate() is None:
            return
        # The CLI reads a file; the exact config is copied into the run directory anyway.
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(self.editor.toPlainText())
            config_file = f.name

        self._run = None
        self._stdout_buffer = ""
        self.log.clear()
        self.curve.setData([], [])

        proc = QProcess(self)
        proc.setWorkingDirectory(str(ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._read_output)
        proc.finished.connect(self._on_finished)
        self._process = proc
        proc.start(sys.executable, ["-u", "-m", "src.ml.train", config_file])

        self.status.setText("Starting…")
        self.launch_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._timer.start(1000)

    def stop(self) -> None:
        """Ask the training process to stop after the current iteration (it still saves weights)."""
        if self._run is not None:
            self._run.request_stop()
            self.status.setText(f"Stopping {self._run.name}…")
        elif self._process is not None:
            self._process.kill()  # not started a run yet: nothing to save
        self.stop_btn.setEnabled(False)

    def shutdown(self, timeout_ms: int = 10_000) -> None:
        """Called when the window closes: stop gracefully, kill if it takes too long."""
        if self._process is None or not self.is_running():
            return
        self.stop()
        if not self._process.waitForFinished(timeout_ms):
            self._process.kill()
            self._process.waitForFinished(2000)

    # ----- internals -----
    def _set_validation(self, ok: bool, text: str) -> None:
        colour = "#2e7d32" if ok else "#c62828"
        self.validation.setStyleSheet(f"color: {colour};")
        self.validation.setText(text)

    def _load_template(self) -> None:
        path: Path | None = self.template_combo.currentData()
        if path is not None and path.exists():
            self.editor.setPlainText(path.read_text())

    def _save_as(self) -> None:
        name, _ = QFileDialog.getSaveFileName(self, "Save config", str(CONFIG.configs_dir), "TOML (*.toml)")
        if name:
            Path(name).write_text(self.editor.toPlainText())
            self.refresh_templates()
            self.template_combo.setCurrentText(Path(name).name)

    def _read_output(self) -> None:
        assert self._process is not None
        self._stdout_buffer += bytes(self._process.readAllStandardOutput().data()).decode(errors="replace")
        *lines, self._stdout_buffer = self._stdout_buffer.split("\n")
        for line in lines:
            line = line.rstrip("\r")
            self.log.appendPlainText(line)
            if line.startswith("RUN_DIR ") and self._run is None:
                self._run = RunDir(Path(line.removeprefix("RUN_DIR ")))
                self.status.setText(f"Training {self._run.name}")
                self.run_started.emit(self._run.path)

    def _update_curve(self) -> None:
        if self._run is None:
            return
        df = self._run.read_metrics()
        if "train_return" in df.columns:
            self.curve.setData(df["iteration"].to_numpy(), df["train_return"].to_numpy())

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        self._timer.stop()
        self._update_curve()
        self.launch_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self._run is not None:
            state = self._run.read_status().state
            self.status.setText(f"{self._run.name}: {state} (exit code {exit_code})")
            self.run_finished.emit(self._run.path)
        else:
            self.status.setText(f"Process exited with code {exit_code} before starting a run")
