from qtpy.QtWidgets import QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rust Stuff GUI")
        self.init_ui()

    def init_ui(self):
        label = QLabel("Hello from Rust Stuff GUI!")
        self.setCentralWidget(label)
    