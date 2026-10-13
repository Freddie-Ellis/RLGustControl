"""A live log window for a child process (the learner or the actor). Closing the window only hides it; the process
keeps running. Launch either a command line (`start`) or a Python function in a fresh interpreter (`run`).

Each line is prefixed with the seconds since launch. Numbers are lightly highlighted, stderr is red, and notes from
the window itself (start, exit) are grey italics. The header gives the process state, PID, elapsed time and line
count, and optionally the latest value of something parsed from the output (e.g. the actor's episode return) with a
small trace of its history.
"""

import html
import pickle
import re
import sys
import tempfile
import time
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pyqtgraph as pg
from qtpy.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

type Channel = Literal["OUT", "ERR", "SYS"]

# Dark and high contrast, but restrained: colour only where it carries meaning.
BG, PANEL, RULE = "#000000", "#0b0b0b", "#2a2a2a"
DIM, TEXT = "#8c8c8c", "#e8e8e8"
NUMBER, RED, GREEN = "#a9c8f0", "#ff6e6e", "#9fd49f"

ERROR_WORDS = re.compile(r"\b(?:Traceback|Error|Exception|failed|not written)\b")
NUMBER_TOKEN = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b")
MAX_LINES = 5000
TICK_MS = 500
SERIF = "Cambria"
MONO = "Consolas"


@dataclass(frozen=True)
class Readout:
    """A value picked out of the output for the header, e.g. Readout("return", r"return\\s+(\\S+)")."""
    label: str
    pattern: str  # its first group is the value
    sparkline: bool = False  # also plot the recent history (the value must be numeric)


def _escape(text: str) -> str:
    """HTML-escape and keep runs of spaces, which the processes use to align columns."""
    return html.escape(text).replace(" ", "&nbsp;")


def _span(text: str, colour: str, italic: bool = False) -> str:
    style = f"color:{colour};" + ("font-style:italic;" if italic else "")
    return f'<span style="{style}">{_escape(text)}</span>'


def _highlight(text: str, base: str) -> str:
    """`base` colour, with the numbers picked out."""
    out: list[str] = []
    end = 0
    for m in NUMBER_TOKEN.finditer(text):
        out.append(_span(text[end:m.start()], base))
        out.append(_span(m.group(), NUMBER))
        end = m.end()
    out.append(_span(text[end:], base))
    return "".join(out)


def _short(value: object) -> str:
    """repr for simple values, just the type for big ones (a whole RunConfig would fill the line)."""
    if isinstance(value, (bool, int, float, str, Path, type(None))) or (isinstance(value, list) and not value):
        return repr(value) if not isinstance(value, Path) else f"Path({str(value)!r})"
    return f"<{type(value).__name__}>"


def _hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class TelemetryConsole(QWidget):
    """Runs one child process and shows its stdout and stderr live, in a window of its own."""

    line_received = Signal(str)  # every stdout line, for parsing elsewhere (e.g. RUN_DIR)
    finished = Signal(int)  # exit code

    def __init__(self, name: str, readout: Readout | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)  # a popup even with a parent (the parent only owns its lifetime)
        self.setWindowTitle(f"{name} — process log")
        self.resize(950, 450)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.name = name
        self._readout = readout
        self._readout_re = re.compile(readout.pattern) if readout else None
        self._process: QProcess | None = None
        self._buffers: dict[Channel, str] = {"OUT": "", "ERR": ""}
        self._t0 = time.monotonic()
        self._n_lines = 0
        self._trace: deque[float] = deque(maxlen=300)
        self._state = "idle"
        self._state_colour = DIM

        # --- header: title and state on the left, readout and controls on the right ---
        self.title = QLabel(name)
        self.title.setFont(QFont(SERIF, 13))
        self.status = QLabel()
        self.status.setFont(QFont(SERIF, 10))
        self.readout_label = QLabel()
        self.readout_label.setFont(QFont(SERIF, 11))
        self.readout_label.setVisible(readout is not None)

        self.spark: pg.PlotWidget | None = None
        self._spark_curve: pg.PlotDataItem | None = None
        if readout is not None and readout.sparkline:
            self.spark = pg.PlotWidget(background=PANEL)
            self.spark.setFixedSize(140, 28)
            self.spark.hideAxis("left")
            self.spark.hideAxis("bottom")
            self.spark.setMouseEnabled(False, False)
            self.spark.hideButtons()
            self._spark_curve = self.spark.plot([], [], pen=pg.mkPen(TEXT, width=1.0))

        self.follow = QCheckBox("Follow output")
        self.follow.setChecked(True)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        for w in (self.follow, clear):
            w.setFont(QFont(SERIF, 10))

        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(self.title)
        titles.addWidget(self.status)

        header = QFrame()
        header.setObjectName("header")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.addLayout(titles)
        row.addStretch()
        row.addWidget(self.readout_label)
        if self.spark is not None:
            row.addWidget(self.spark)
        row.addSpacing(16)
        row.addWidget(self.follow)
        row.addWidget(clear)

        # --- body ---
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont(MONO, 9))
        self.text.setMaximumBlockCount(MAX_LINES)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.text, 1)
        self.setStyleSheet(f"""
            TelemetryConsole {{ background: {BG}; }}
            QFrame#header {{ background: {PANEL}; border: none; border-bottom: 1px solid {RULE}; }}
            QFrame#header QLabel, QCheckBox {{ color: {TEXT}; background: transparent; }}
            QPlainTextEdit {{ background: {BG}; color: {TEXT}; border: none;
                              selection-background-color: #3a3a3a; selection-color: #ffffff; padding: 8px; }}
            QScrollBar {{ background: {BG}; width: 10px; height: 10px; }}
            QScrollBar::handle {{ background: {RULE}; min-height: 20px; min-width: 20px; }}
            QScrollBar::handle:hover {{ background: {DIM}; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
            QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
            QPushButton {{ color: {TEXT}; background: transparent; border: 1px solid {RULE}; padding: 3px 12px; }}
            QPushButton:hover {{ border-color: {DIM}; }}
        """)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_header)
        self._refresh_header()

    def popup(self) -> None:
        """Show the window (again) and bring it to the front."""
        self.show()
        self.raise_()
        self.activateWindow()

    # ----- process -----
    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    def start(self, program: str, args: list[str], cwd: str, description: str | None = None) -> None:
        """Launch the process (ignored if one is already running). `description` replaces the command line in the log."""
        if self.is_running():
            self.system("already running", RED)
            return
        self._t0 = time.monotonic()
        self._trace.clear()
        self._buffers = {"OUT": "", "ERR": ""}
        if self.text.blockCount() > 1:
            self.text.appendHtml(_span("", DIM))
        self.system(f"{self.name}, started {datetime.now():%d %b %Y, %H:%M:%S}")
        self.system(description or "python " + " ".join(args))

        proc = QProcess(self)
        proc.setWorkingDirectory(cwd)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(lambda: self._read("OUT"))
        proc.readyReadStandardError.connect(lambda: self._read("ERR"))
        proc.started.connect(self._on_started)
        proc.errorOccurred.connect(self._on_error)
        proc.finished.connect(self._on_finished)
        self._process = proc
        self._set_state("starting", DIM)
        self._timer.start(TICK_MS)
        proc.start(program, args)

    def run(self, fn: Callable[..., object], /, *args: object, cwd: str = ".", **kwargs: object) -> None:
        """Call `fn(*args, **kwargs)` in a new Python process and show its output. `fn` must be a top-level function
        (it is pickled by reference) and the arguments picklable; both are checked here, before anything starts."""
        if self.is_running():
            self.system("already running", RED)
            return
        try:
            job = pickle.dumps((fn, args, kwargs))
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            self.system(f"cannot send {getattr(fn, '__qualname__', fn)!r} to a new process: {e}", RED)
            return
        with tempfile.NamedTemporaryFile("wb", suffix=".pkl", delete=False) as f:
            f.write(job)  # the child deletes it once loaded
        call = ", ".join([*(_short(a) for a in args), *(f"{k}={_short(v)}" for k, v in kwargs.items())])
        self.start(sys.executable, ["-u", "-m", "src.ui.utils.run_callable", f.name], cwd,
                   description=f"{fn.__module__}.{fn.__qualname__}({call})")

    def kill(self, wait_ms: int = 2000) -> None:
        if self._process is not None and self.is_running():
            self.system("killed", RED)
            self._process.kill()
            self._process.waitForFinished(wait_ms)

    def wait(self, timeout_ms: int) -> bool:
        """Block until the process exits. True if it did."""
        return self._process is None or not self.is_running() or self._process.waitForFinished(timeout_ms)

    # ----- output -----
    def system(self, message: str, colour: str = DIM) -> None:
        """A note from the window itself, not the process."""
        self._append("SYS", _span(message, colour, italic=True))

    def clear(self) -> None:
        self.text.clear()

    def _read(self, channel: Channel) -> None:
        if self._process is None:
            return
        raw = self._process.readAllStandardOutput() if channel == "OUT" else self._process.readAllStandardError()
        self._buffers[channel] += bytes(raw.data()).decode("utf-8", errors="replace")
        *lines, self._buffers[channel] = self._buffers[channel].split("\n")
        for line in lines:
            self._emit_line(channel, line.rstrip("\r"))

    def _flush(self) -> None:
        """Whatever is left without a trailing newline when the process exits."""
        channels: tuple[Channel, ...] = ("OUT", "ERR")
        for channel in channels:
            self._read(channel)
            if self._buffers[channel]:
                self._emit_line(channel, self._buffers[channel])
                self._buffers[channel] = ""

    def _emit_line(self, channel: Channel, line: str) -> None:
        if not line.strip():
            return
        is_error = channel == "ERR" or ERROR_WORDS.search(line) is not None
        self._append(channel, _span(line, RED) if is_error else _highlight(line, TEXT))
        self._n_lines += 1
        if channel == "OUT":
            self._parse_readout(line)
            self.line_received.emit(line)

    def _append(self, channel: Channel, body: str) -> None:
        """One line: elapsed seconds in the gutter, a marker for stderr, then the text."""
        gutter = _span(f"{time.monotonic() - self._t0:9.2f}", DIM)
        marker = _span(" ! ", RED) if channel == "ERR" else _span("   ", DIM)
        bar = self.text.verticalScrollBar()
        held = bar.value()
        self.text.appendHtml(gutter + marker + body)
        if self.follow.isChecked():
            bar.setValue(bar.maximum())
            self.text.horizontalScrollBar().setValue(0)  # a long line would otherwise scroll the view sideways
        else:
            bar.setValue(held)

    def _parse_readout(self, line: str) -> None:
        if self._readout is None or self._readout_re is None:
            return
        m = self._readout_re.search(line)
        if m is None:
            return
        value = m.group(1)
        self.readout_label.setText(
            f'<i style="color:{DIM}">{html.escape(self._readout.label)}</i>'
            f'<span style="color:{DIM}"> = </span>'
            f'<span style="color:{TEXT}">{html.escape(value)}</span>'
        )
        if self._spark_curve is not None:
            try:
                self._trace.append(float(value))
            except ValueError:
                return
            self._spark_curve.setData(list(self._trace))

    # ----- header -----
    def _set_state(self, state: str, colour: str) -> None:
        self._state, self._state_colour = state, colour
        self._refresh_header()

    def _refresh_header(self) -> None:
        running = self.is_running()
        parts = [f'<span style="color:{self._state_colour}">{self._state}</span>']
        if running and self._process is not None and self._process.processId():
            parts.append(f"pid {self._process.processId()}")
        if self._process is not None:
            parts.append(f"elapsed {_hms(time.monotonic() - self._t0)}")
        parts.append(f"{self._n_lines} lines")
        self.status.setText(f'<span style="color:{DIM}">{" · ".join(parts)}</span>')

    def _on_started(self) -> None:
        self._set_state("running", GREEN)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._set_state("failed to start", RED)
            self.system("failed to start", RED)
            self._timer.stop()

    def _on_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        self._flush()
        self._timer.stop()
        crashed = status == QProcess.ExitStatus.CrashExit or exit_code != 0
        elapsed = _hms(time.monotonic() - self._t0)
        self._set_state(f"exited ({exit_code})", RED if crashed else DIM)
        self.system(f"exited with code {exit_code} after {elapsed}", RED if crashed else DIM)
        self.finished.emit(exit_code)
