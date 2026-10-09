"""Launch the experiment UI"""

import sys

from qtpy.QtWidgets import QApplication

from src.ui.mainwindow import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
