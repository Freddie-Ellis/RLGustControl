"""Live view of an actor + learner session, read from their shared folder.

Left, one point per episode (x = episode): return, control loop timing, and which weight version the actor used.
Right: the latest episode in detail (Cm over time, every tap pressure and surface command as heatmaps) and the learner's metrics.
Everything comes from files the actor and learner already write, so it works for a session on another machine too.
A learner and actor launched from here each get a live console window.
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QRectF, Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.config import ROOT
from src.ml.distributed import actor, learner
from src.ml.distributed.files import stop_path
from src.ml.distributed.session import EpisodeSummary, SessionReader
from src.ml.run_config import RunConfig
from src.ui.utils.dynamic_vb import DynamicViewBox
from src.ui.utils.logger import RED, Readout, TelemetryConsole

DEFAULT_SHARED = Path(r"C:\rl_shared\session1")
POLL_MS = 1000
STOP_WAIT_MS = 30_000  # on close: time for the actor to finish its episode and the learner its chunk after STOP
KIND_STYLE: dict[str, tuple[tuple[int, int, int], int]] = {  # colour, marker size
    "warmup": ((140, 140, 140), 5),
    "train": ((80, 170, 240), 5),
    "eval": ((255, 190, 60), 9),
}
GOOD, BAD, IDLE = "#4caf50", "#e57373", "#9e9e9e"


def _dot(colour: str, text: str) -> str:
    return f'<span style="color:{colour}">●</span> {text}'


class SessionTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.reader: SessionReader | None = None

        # --- top: folder + status + stop ---
        self.folder = QLineEdit(str(DEFAULT_SHARED))
        launch_learner = QPushButton("Launch learner")
        launch_learner.clicked.connect(self._launch_learner)
        launch_actor = QPushButton("Launch actor")
        launch_actor.clicked.connect(self._launch_actor)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        logs = QPushButton("Logs")
        logs.setToolTip("Show the learner and actor consoles")
        logs.clicked.connect(self._show_consoles)
        watch = QPushButton("Watch")
        watch.clicked.connect(self._watch)
        self.stop_btn = QPushButton("Stop session")
        self.stop_btn.clicked.connect(self._toggle_stop)
        self.stop_btn.setEnabled(False)

        top = QHBoxLayout()
        top.addWidget(QLabel("Shared folder"))
        top.addWidget(self.folder, 1)
        for b in (launch_learner, launch_actor, logs, browse, watch, self.stop_btn):
            top.addWidget(b)

        self.actor_status = QLabel(_dot(IDLE, "Actor: not watching"))
        self.learner_status = QLabel(_dot(IDLE, "Learner: -"))
        self.progress = QLabel("")
        status = QHBoxLayout()
        for lbl in (self.actor_status, self.learner_status, self.progress):
            lbl.setTextFormat(Qt.TextFormat.RichText)
            status.addWidget(lbl)
            status.addSpacing(24)
        status.addStretch()

        # --- left: per-episode plots, x-linked ---
        self.episode_plots = pg.GraphicsLayoutWidget()
        self.p_ret = self._plot(self.episode_plots, 0, "return")
        self.p_ret.addLegend(offset=(-10, 10))
        self.p_dt = self._plot(self.episode_plots, 1, "loop period [ms]")
        self.p_ver = self._plot(self.episode_plots, 2, "weights version")
        self.p_ver.setLabel("bottom", "episode")
        for p in (self.p_dt, self.p_ver):
            p.setXLink(self.p_ret)
        self.ret_items = {
            kind: self.p_ret.plot([], [], pen=None, symbol="o", symbolSize=size, symbolPen=None,
                                  symbolBrush=pg.mkBrush(*colour), name=kind)
            for kind, (colour, size) in KIND_STYLE.items()
        }
        self.dt_mean = self.p_dt.plot([], [], pen=None, symbol="o", symbolSize=4, symbolPen=None,
                                      symbolBrush=pg.mkBrush(80, 170, 240), name="mean")
        self.dt_max = self.p_dt.plot([], [], pen=None, symbol="t", symbolSize=6, symbolPen=None,
                                     symbolBrush=pg.mkBrush(229, 115, 115), name="max")
        self.dt_target = pg.InfiniteLine(angle=0, pen=pg.mkPen("w", style=Qt.PenStyle.DashLine))
        self.p_dt.addItem(self.dt_target, ignoreBounds=True)  # reference lines must not set the y-range
        self.ver_used = self.p_ver.plot([], [], pen=None, symbol="o", symbolSize=4, symbolPen=None,
                                        symbolBrush=pg.mkBrush(80, 170, 240))
        self.ver_latest = pg.InfiniteLine(angle=0, pen=pg.mkPen((255, 190, 60), style=Qt.PenStyle.DashLine),
                                          label="published v{value:.0f}", labelOpts={"position": 0.05})
        self.p_ver.addItem(self.ver_latest, ignoreBounds=True)

        # --- right: latest episode + learner ---
        self.latest_title = QLabel("Latest episode: -")
        self.detail_plots = pg.GraphicsLayoutWidget()
        self.p_cm = self._plot(self.detail_plots, 0, "Cm")
        self.p_taps = self.detail_plots.addPlot(row=1, col=0)
        self.p_taps.setLabel("left", "tap")
        self.p_taps.setXLink(self.p_cm)
        self.p_act = self.detail_plots.addPlot(row=2, col=0)
        self.p_act.setLabel("left", "surface")
        self.p_act.setLabel("bottom", "time [s]")
        self.p_act.setXLink(self.p_cm)
        self.cm_curve = self.p_cm.plot([], [], pen=pg.mkPen((80, 170, 240), width=1.5))
        self.p_cm.addItem(pg.InfiniteLine(angle=0, pos=0.0, pen=pg.mkPen("w", width=1)), ignoreBounds=True)
        self.trip_lines = [
            pg.InfiniteLine(angle=0, pen=pg.mkPen((229, 115, 115), style=Qt.PenStyle.DashLine)) for _ in range(2)
        ]
        for line in self.trip_lines:
            self.p_cm.addItem(line, ignoreBounds=True)  # else +-cm_trip squashes the Cm trace
            line.hide()
        self.act_image = pg.ImageItem()
        self.p_act.addItem(self.act_image)
        colormap = pg.colormap.get("CET-D1")
        self.act_bar = pg.ColorBarItem(values=(-1.0, 1.0), colorMap=colormap, interactive=False, label="command")
        self.act_bar.setImageItem(self.act_image, insert_in=self.p_act)
        self.taps_image = pg.ImageItem()
        self.p_taps.addItem(self.taps_image)
        self.taps_bar = pg.ColorBarItem(values=(-1.0, 1.0), colorMap=colormap, interactive=False, label="Pa")
        self.taps_bar.setImageItem(self.taps_image, insert_in=self.p_taps)
        self.taps_mode = QComboBox()
        self.taps_mode.addItems(["fluctuation", "absolute"])
        self.taps_mode.currentTextChanged.connect(lambda _: self.reader is not None and self._draw_latest())
        detail_row = QHBoxLayout()
        detail_row.addWidget(self.latest_title, 1)
        detail_row.addWidget(QLabel("Pressures"))
        detail_row.addWidget(self.taps_mode)

        self.learner_metric = QComboBox()
        self.learner_metric.currentTextChanged.connect(lambda _: self._draw_learner())
        self.p_learn = pg.PlotWidget(viewBox=DynamicViewBox())
        self.p_learn.showGrid(x=True, y=True, alpha=0.3)
        self.p_learn.setLabel("bottom", "SAC updates")
        self.learn_curve = self.p_learn.plot([], [], pen=pg.mkPen((255, 190, 60), width=1.5), symbol="o", symbolSize=4)
        learner_row = QHBoxLayout()
        learner_row.addWidget(QLabel("Learner metric"))
        learner_row.addWidget(self.learner_metric, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addLayout(detail_row)
        right_layout.addWidget(self.detail_plots, 4)
        right_layout.addLayout(learner_row)
        right_layout.addWidget(self.p_learn, 2)

        split = QSplitter()
        split.addWidget(self.episode_plots)
        split.addWidget(right)
        split.setSizes([700, 600])

        # --- console windows of the processes launched from here (shown on launch) ---
        self.learner_console = TelemetryConsole("Learner", Readout("weights", r"published (v\d+)"), parent=self)
        self.actor_console = TelemetryConsole(
            "Actor", Readout("return", r"return\s+(\S+)", sparkline=True), parent=self
        )

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(status)
        layout.addWidget(split, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(POLL_MS)

    # ----- setup -----
    @staticmethod
    def _plot(widget: pg.GraphicsLayoutWidget, row: int, label: str) -> pg.PlotItem:
        p: pg.PlotItem = widget.addPlot(row=row, col=0, viewBox=DynamicViewBox())
        p.setLabel("left", label)
        p.showGrid(x=True, y=True, alpha=0.3)
        return p

    def _launch_learner(self) -> None:
        """Run the learner in a new process on the shared folder in the box, with learner.py's settings."""
        shared = self._shared_for(self.learner_console)
        if shared is not None:
            cfg = RunConfig.from_toml(ROOT / learner.CONFIG_FILE)
            if learner.RUN_NAME:
                cfg = cfg.model_copy(update={"name": learner.RUN_NAME})
            self.learner_console.run(
                learner.run_learner, cfg, shared,
                seed_from=learner.SEED_FROM, max_updates=learner.MAX_UPDATES, cwd=str(ROOT),
            )

    def _launch_actor(self) -> None:
        """Run the actor in a new process on the shared folder in the box, with actor.py's settings."""
        shared = self._shared_for(self.actor_console)
        if shared is not None:
            self.actor_console.run(
                actor.run_actor, shared,
                n_envs=actor.N_ENVS, eval_every=actor.EVAL_EVERY, max_episodes=actor.MAX_EPISODES, cwd=str(ROOT),
            )

    def _shared_for(self, console: TelemetryConsole) -> Path | None:
        """The shared folder to launch on, or None (with the reason in the console) if launching would be pointless."""
        shared = Path(self.folder.text())
        console.popup()
        if stop_path(shared).exists():  # it would exit straight away
            console.system(f"STOP is set in {shared}: clear it first", RED)
            return None
        return shared

    def _show_consoles(self) -> None:
        self.learner_console.popup()
        self.actor_console.popup()

    def shutdown(self) -> bool:
        """Called when the window closes. Launched processes write to pipes owned by this window, so they cannot
        outlive it: ask to stop them (gracefully via STOP, killed if too slow). False = the user cancelled."""
        running = [c for c in (self.learner_console, self.actor_console) if c.is_running()]
        if not running:
            return True
        names = " and ".join(c.name.lower() for c in running)
        answer = QMessageBox.question(self, "Session running", f"Stop the {names} and close?")
        if answer != QMessageBox.StandardButton.Yes:
            return False
        stop_path(Path(self.folder.text())).touch()
        for console in running:
            if not console.wait(STOP_WAIT_MS):
                console.kill()
        return True

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Shared folder", self.folder.text())
        if path:
            self.folder.setText(path)
            self._watch()

    def _watch(self) -> None:
        """Start reading the folder in the box from scratch."""
        self.reader = SessionReader(Path(self.folder.text()))
        self.stop_btn.setEnabled(True)
        self.learner_metric.clear()
        self._tick()

    # ----- updates -----
    def _tick(self) -> None:
        if self.reader is None or not self.isVisible():
            return
        if self.reader.poll():
            self._draw_episodes()
            self._draw_latest()
        self._draw_status()
        self._draw_learner()

    def _draw_status(self) -> None:
        r = self.reader
        assert r is not None
        if r.cfg is None:
            self.actor_status.setText(_dot(IDLE, f"Waiting for config.json in {r.shared}"))
            return
        since = r.seconds_since_last_episode()
        expected = r.episode_seconds() or 0.0
        if r.stop_requested():
            actor = _dot(BAD, "Actor: STOP requested")
        elif since is None:
            actor = _dot(IDLE, "Actor: no episodes yet")
        elif since < 2 * expected + 5:
            actor = _dot(GOOD, f"Actor: running, last episode {since:.0f} s ago")
        else:
            actor = _dot(IDLE, f"Actor: idle, last episode {since / 60:.1f} min ago")
        self.actor_status.setText(actor)

        status = r.learner_status()
        version = r.latest_version()
        state = status.state if status else "not found"
        colour = GOOD if state == "running" else (BAD if state == "failed" else IDLE)
        self.learner_status.setText(_dot(colour, f"Learner: {state}, weights v{version if version else 0}"))

        warm = r.warmup_transitions
        warm_text = "" if warm is None else (
            "warmup done" if r.transitions >= warm else f"warmup {r.transitions}/{warm}"
        )
        self.progress.setText(f"{len(r.episodes)} episodes · {r.transitions} transitions · {warm_text}")
        self.stop_btn.setText("Clear STOP" if r.stop_requested() else "Stop session")

    def _draw_episodes(self) -> None:
        r = self.reader
        assert r is not None
        eps = r.episodes
        for kind, item in self.ret_items.items():
            pts = [e for e in eps if e.kind == kind]
            item.setData([e.episode_id for e in pts], [e.ret for e in pts])
        ids = np.array([e.episode_id for e in eps], dtype=float)
        self.dt_mean.setData(ids, [1000 * e.dt_mean for e in eps])
        self.dt_max.setData(ids, [1000 * e.dt_max for e in eps])
        if r.target_dt is not None:
            self.dt_target.setValue(1000 * r.target_dt)
        used = [e for e in eps if e.version >= 0]
        self.ver_used.setData([e.episode_id for e in used], [e.version for e in used])
        latest = r.latest_version()
        self.ver_latest.setVisible(latest is not None)
        if latest is not None:
            self.ver_latest.setValue(latest)

    def _draw_latest(self) -> None:
        r = self.reader
        assert r is not None
        d = r.latest
        if d is None:
            return
        summary: EpisodeSummary = r.episodes[-1]
        parts = [f"Latest episode #{d.meta.episode_id}", summary.kind, f"weights v{d.meta.policy_version}"]
        if summary.turbulence is not None:
            parts.append(f"turbulence {summary.turbulence:.2f}°")
        if summary.cm_rms is not None:
            parts.append(f"Cm rms {summary.cm_rms:.4f}")
        parts.append(f"return {summary.ret:.1f}")
        self.latest_title.setText(" · ".join(parts))

        if d.cm is not None:
            self.cm_curve.setData(d.t.numpy(), d.cm.numpy())
        trip = r.cm_trip
        for line, sign in zip(self.trip_lines, (1.0, -1.0)):
            line.setVisible(trip is not None)
            if trip is not None:
                line.setValue(sign * trip)

        act = d.act.numpy()  # [T, surfaces]; ImageItem's first axis is x, so time runs along x
        self.act_image.setImage(act, levels=(-1.0, 1.0), autoLevels=False)
        dt = float(d.t[0]) if len(d.t) else 1.0
        self.act_image.setRect(QRectF(0.0, 0.0, act.shape[0] * dt, act.shape[1]))
        self.p_act.setLimits(yMin=0, yMax=act.shape[1])

        self.p_taps.setVisible(d.taps is not None)
        if d.taps is not None:
            taps = d.taps.numpy()  # [T, n_taps] Pa
            if self.taps_mode.currentText() == "fluctuation":
                # Each tap minus its episode mean: the steady Cp distribution would otherwise hide the gusts.
                taps = taps - taps.mean(axis=0, keepdims=True)
                span = float(np.percentile(np.abs(taps), 99)) or 1.0  # robust to one noisy spike
                levels = (-span, span)
            else:
                levels = (float(taps.min()), float(taps.max()))
            self.taps_image.setImage(taps, levels=levels, autoLevels=False)
            self.taps_bar.setLevels(levels)
            self.taps_image.setRect(QRectF(0.0, 0.0, taps.shape[0] * dt, taps.shape[1]))
            self.p_taps.setLimits(yMin=0, yMax=taps.shape[1])

    def _draw_learner(self) -> None:
        if self.reader is None:
            return
        df = self.reader.learner_metrics()
        if df.is_empty():
            return
        columns = [c for c in df.columns if c != "iteration"]
        if self.learner_metric.count() != len(columns):
            current = self.learner_metric.currentText() or "critic_loss"
            self.learner_metric.blockSignals(True)
            self.learner_metric.clear()
            self.learner_metric.addItems(columns)
            self.learner_metric.setCurrentText(current if current in columns else columns[0])
            self.learner_metric.blockSignals(False)
        metric = self.learner_metric.currentText()
        if metric in df.columns:
            y = df[metric].cast(float).to_numpy()
            x = df["iteration"].to_numpy()
            keep = np.isfinite(y)  # eval_return is NaN for versions with no eval episode
            self.learn_curve.setData(x[keep], y[keep])
            self.p_learn.setLabel("left", metric)

    def _toggle_stop(self) -> None:
        if self.reader is None:
            return
        if self.reader.stop_requested():
            self.reader.clear_stop()
        else:
            self.reader.request_stop()
        self._draw_status()
