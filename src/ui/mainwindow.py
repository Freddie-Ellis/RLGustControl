from pathlib import Path

from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QMainWindow, QTabWidget

from src.ui.tabs.data_tab import DataTab
from src.ui.tabs.rollout_tab import RolloutTab
from src.ui.tabs.runs_tab import RunsTab
from src.ui.tabs.train_tab import TrainTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RL Gust Control")
        self.resize(1300, 820)

        self.runs = RunsTab()
        self.rollout = RolloutTab()
        self.train = TrainTab()
        self.data = DataTab()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.runs, "Runs")
        self.tabs.addTab(self.rollout, "Rollout")
        self.tabs.addTab(self.train, "Train")
        self.tabs.addTab(self.data, "Data")
        self.setCentralWidget(self.tabs)

        self.runs.replay_requested.connect(self._replay)
        self.runs.template_requested.connect(self._use_as_template)
        self.train.run_started.connect(lambda _: self.runs.refresh())
        self.train.run_finished.connect(self._on_run_finished)

    def _replay(self, path: Path) -> None:
        self.rollout.refresh_runs()
        self.rollout.select_run(path)
        self.tabs.setCurrentWidget(self.rollout)
        self.rollout.run_rollout()

    def _use_as_template(self, path: Path) -> None:
        self.train.load_from_run(path)
        self.tabs.setCurrentWidget(self.train)

    def _on_run_finished(self, _path: Path) -> None:
        self.runs.refresh()
        self.rollout.refresh_runs()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.train.shutdown()
        super().closeEvent(event)
