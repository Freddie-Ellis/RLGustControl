"""Pop-up of the Session tab for setting the targets the policy holds (the loads it tracks, e.g. lift, pitch, roll).

Display only: the tracked loads, their trained ranges, the timing and what is in force come from the SessionReader,
and the targets are written (and checked) by it. Nothing here decides anything.
"""

from datetime import datetime

from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from src.ml.distributed.files import TargetRequest, TargetScope
from src.ml.distributed.session import SessionReader

LOAD_LABELS: dict[str, str] = {"C_L": "Lift C_L", "C_m": "Pitch C_m", "C_l": "Roll C_l"}


class TargetsDialog(QDialog):
    def __init__(self, reader: SessionReader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Targets")
        self.reader = reader
        tracking = reader.tracking()
        current = reader.target_request()
        preset = current.targets if current is not None else reader.config_eval_targets()

        when = QLabel(self._timing_text())
        when.setWordWrap(True)

        # One row per tracked load: pin it, and to what. Spin boxes only go over the trained range.
        loads = QGroupBox("Targets (unticked: drawn at random from the range, as in training)")
        grid = QGridLayout(loads)
        self.pins: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        for row, (name, load) in enumerate(tracking.items()):
            low, high = load.target
            pin = QCheckBox(LOAD_LABELS.get(name, name))
            value = QDoubleSpinBox()
            value.setDecimals(4)
            value.setRange(low, high)
            value.setSingleStep(max((high - low) / 50, 1e-4))
            value.setValue(preset.get(name, (low + high) / 2))
            pin.setChecked(name in preset)
            value.setEnabled(pin.isChecked() and high > low)
            pin.toggled.connect(lambda on, v=value, fixed=high <= low: v.setEnabled(on and not fixed))
            trained = f"trained on {low:g}" if high <= low else f"trained range {low:g} to {high:g}"
            grid.addWidget(pin, row, 0)
            grid.addWidget(value, row, 1)
            grid.addWidget(QLabel(trained), row, 2)
            self.pins[name] = (pin, value)

        scope_box = QGroupBox("Which episodes hold them")
        scope_layout = QVBoxLayout(scope_box)
        self.eval_only = QRadioButton("Evaluation and baseline episodes only. Training keeps drawing random targets, "
                                      "so the policy keeps learning every target (recommended while training)")
        self.every = QRadioButton("Every episode, training too. The policy then only practises these targets")
        group = QButtonGroup(self)
        for button in (self.eval_only, self.every):
            group.addButton(button)
            scope_layout.addWidget(button)
        (self.every if current is not None and current.scope == "all" else self.eval_only).setChecked(True)

        now = QLabel(self._in_force_text(current))
        now.setWordWrap(True)

        buttons = QDialogButtonBox()
        apply = buttons.addButton("Apply from next episode", QDialogButtonBox.ButtonRole.AcceptRole)
        reset = buttons.addButton("Back to the config's", QDialogButtonBox.ButtonRole.ResetRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        apply.clicked.connect(self._apply)
        reset.clicked.connect(self._reset)
        buttons.rejected.connect(self.reject)
        reset.setToolTip("Delete the targets set here: evaluation and baseline episodes hold the config's "
                         f"eval_targets ({reader.config_eval_targets() or 'none: drawn at random'}), training "
                         "episodes draw theirs")

        layout = QVBoxLayout(self)
        layout.addWidget(when)
        layout.addWidget(loads)
        layout.addWidget(scope_box)
        layout.addWidget(now)
        layout.addWidget(buttons)

    def _timing_text(self) -> str:
        """When a change kicks in: the actor reads the targets as it starts each episode."""
        episode = self.reader.episode_seconds()
        every = self.reader.eval_every
        text = ("<b>When they kick in:</b> the actor reads the targets as it starts each episode, so they apply from "
                "its <b>next</b> episode. The episode running now finishes with the targets it started with.")
        if episode is not None:
            text += f" Episodes take about {episode:.1f} s each (settling included), so that is within ~{episode:.1f} s."
            if every:
                text += (f" With 'evaluation episodes only', they show in the next evaluation episode: every "
                         f"{every}th episode, about every {every * episode:.0f} s.")
        return text

    def _in_force_text(self, current: TargetRequest | None) -> str:
        set_here = ("Set here: none (the config's targets apply)" if current is None else
                    f"Set here at {current.written:%H:%M:%S}: {_fmt(current.targets)} for "
                    f"{'every episode' if current.scope == 'all' else 'eval and baseline episodes'}")
        latest = self.reader.latest_targets()
        ran = ("" if latest is None else
               f"<br>Latest episode (#{latest[0]}) ran with: {_fmt(latest[1])}"
               + (" (set)" if latest[2] else " (drawn at random)"))
        return set_here + ran

    def _apply(self) -> None:
        targets = {name: value.value() for name, (pin, value) in self.pins.items() if pin.isChecked()}
        scope: TargetScope = "all" if self.every.isChecked() else "eval"
        try:
            self.reader.set_target_request(
                TargetRequest(targets=targets, scope=scope, written=datetime.now().astimezone()))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Targets not set", str(e))
            return
        self.accept()

    def _reset(self) -> None:
        try:
            self.reader.set_target_request(None)
        except OSError as e:
            QMessageBox.warning(self, "Targets not cleared", str(e))
            return
        self.accept()


def _fmt(targets: dict[str, float]) -> str:
    return ", ".join(f"{name} {value:.4g}" for name, value in targets.items()) or "all drawn at random"
