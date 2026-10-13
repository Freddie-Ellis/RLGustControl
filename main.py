import sys

from qtpy.QtWidgets import QApplication

from src.ui.mainwindow import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()

    exit_code: int = app.exec()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()