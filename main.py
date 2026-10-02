import sys

from qtpy.QtWidgets import QApplication

import ruststuff
from src.ui.mainwindow import MainWindow


def main():
    print(ruststuff.step(0.0, 0.0, 1.0))

    app = QApplication([])
    mw = MainWindow()
    mw.show()

    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
