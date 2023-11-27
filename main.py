# Project: Demonstration of backpropagation learning - basic algorithm and selected optimizer
# Author: David Chocholaty <xchoch09@stud.fit.vutbr.cz>
# File: main.py

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
