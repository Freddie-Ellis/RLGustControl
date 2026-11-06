import cProfile
import sys

prfl = cProfile.Profile()
prfl.enable()

from qtpy.QtWidgets import QApplication

from src.ui.mainwindow import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    app.processEvents() 

    prfl.disable()
    prfl.dump_stats("profile.prof") 

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
