import sys

from main_window import MainWindow
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication([])
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()