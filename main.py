import cProfile
import sys

prfl = cProfile.Profile()
prfl.enable()  # before the imports below: loading torch, pyqtgraph and the tabs is most of startup

from qtpy.QtWidgets import QApplication

from src.ui.mainwindow import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    app.processEvents()  # the first paint, so the profile covers everything until the window is up

    prfl.disable()
    prfl.dump_stats("profile.prof")  # before the event loop: nothing after app.exec() runs until the window closes

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
