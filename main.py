from frontend.mainwindowui import MainWindowUI

from PyQt5.QtWidgets import QApplication, QMainWindow

import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui_main_window = MainWindowUI()
        self.ui_main_window.ui_setup(self)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()

    return app.exec_()


if __name__ == "__main__":
    ret_code = main()
    sys.exit(ret_code)
