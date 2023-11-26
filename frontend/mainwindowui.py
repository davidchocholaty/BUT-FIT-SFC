import numpy as np

from engine.nn.mlp import MultilayerPerceptron
from engine.opt.optimizer import Optimizer
from model.constants import PIXELS_PER_IMAGE, HIDDEN_LAYER_SIZE, NB_LABELS, HIDDEN_LAYER_ACT, OUTPUT_LAYER_ACT
from model.train import run, run_init_loss
from model.data_prep import DataPreprocessor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QCheckBox, QHBoxLayout
from PyQt5.QtCore import QRect, QLocale, Qt, QMetaObject
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont, QMovie


class MainWindowUI(object):
    def __init__(self):
        ####################
        #    Qt params     #
        ####################

        self.movie_label = None
        self.amsgrad_line = None
        self.adam_line = None
        self.rmsprop_line = None
        self.noopt_line = None
        self.ax = None
        self.opts_label = None
        self.hpparams_label = None
        self.amsgrad_chbox = None
        self.rmsprop_chbox = None
        self.canvas = None
        self.horizontal_layout = None
        self.horizontal_layout_widget = None
        self.adam_chbox = None
        self.noopt_chbox = None
        self.reset_btn = None
        self.start_btn = None
        self.nb_epochs_validator = None
        self.nb_epochs_edit = None
        self.nb_epochs_label = None
        self.lr_validator = None
        self.lr_edit = None
        self.lr_label = None
        self.widget = None
        self.window_width = 1200
        self.window_height = 600
        self.min_window_width = 1120
        self.min_window_height = 480
        self.label_font = QFont("Arial", 11)
        self.header_font = QFont("Arial", 15)

        self.header_font.setBold(True)

        ####################
        #   Model params   #
        ####################
        self.epochs_done = 0
        self.nb_epochs = 11  # 10 + 1
        self.learning_rate = 0.01

        self.data = DataPreprocessor()
        self.data.create_dataset()

        self.noopt_model = None
        self.rmsprop_model = None
        self.adam_model = None
        self.amsgrad_model = None

        self.noopt_loss = []
        self.rmsprop_loss = []
        self.adam_loss = []
        self.amsgrad_loss = []

    def init_models(self):
        self.nb_epochs = int(self.nb_epochs_edit.text()) + 1
        self.learning_rate = float(self.lr_edit.text())

        if self.noopt_chbox.isChecked():
            self.noopt_model = MultilayerPerceptron(PIXELS_PER_IMAGE,
                                                    [(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_ACT),
                                                     (NB_LABELS, OUTPUT_LAYER_ACT)])
        else:
            self.noopt_model = None

        if self.rmsprop_chbox.isChecked():
            opt = Optimizer.RMSpropOptimizer(params_size_list=[(PIXELS_PER_IMAGE, HIDDEN_LAYER_SIZE),
                                                               HIDDEN_LAYER_SIZE,
                                                               (HIDDEN_LAYER_SIZE, NB_LABELS),
                                                               NB_LABELS])
            self.rmsprop_model = MultilayerPerceptron(PIXELS_PER_IMAGE,
                                                      [(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_ACT),
                                                       (NB_LABELS, OUTPUT_LAYER_ACT)], opt=opt)
        else:
            self.rmsprop_model = None

        if self.adam_chbox.isChecked():
            opt = Optimizer.AdamOptimizer(params_size_list=[(PIXELS_PER_IMAGE, HIDDEN_LAYER_SIZE),
                                                            HIDDEN_LAYER_SIZE,
                                                            (HIDDEN_LAYER_SIZE, NB_LABELS),
                                                            NB_LABELS])
            self.adam_model = MultilayerPerceptron(PIXELS_PER_IMAGE,
                                                   [(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_ACT),
                                                    (NB_LABELS, OUTPUT_LAYER_ACT)], opt=opt)
        else:
            self.adam_model = None

        if self.amsgrad_chbox.isChecked():
            opt = Optimizer.AmsGradOptimizer(params_size_list=[(PIXELS_PER_IMAGE, HIDDEN_LAYER_SIZE),
                                                               HIDDEN_LAYER_SIZE,
                                                               (HIDDEN_LAYER_SIZE, NB_LABELS),
                                                               NB_LABELS])
            self.amsgrad_model = MultilayerPerceptron(PIXELS_PER_IMAGE,
                                                      [(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_ACT),
                                                       (NB_LABELS, OUTPUT_LAYER_ACT)], opt=opt)
        else:
            self.amsgrad_model = None

    def clear_axes(self):
        self.horizontal_layout.removeWidget(self.canvas)
        self.canvas.deleteLater()
        self.canvas = None

    def clear_plot(self):
        self.clear_axes()

        self.canvas = FigureCanvas(Figure(figsize=(5, 3), tight_layout=True))
        self.horizontal_layout.addWidget(self.canvas)

    def update_plot(self):
        xs = range(self.epochs_done)

        self.ax.cla()
        self.ax.set_title('Demonstration of backpropagation using optimizers')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')

        if self.noopt_model:
            self.noopt_line, = self.ax.plot(xs, self.noopt_loss, marker='v', label='No optimizer')
            self.noopt_line.figure.canvas.draw()

        if self.rmsprop_model:
            self.rmsprop_line, = self.ax.plot(xs, self.rmsprop_loss, marker='v', label='RMSprop optimizer')
            self.rmsprop_line.figure.canvas.draw()

        if self.adam_model:
            self.adam_line, = self.ax.plot(xs, self.adam_loss, marker='v', label='Adam optimizer')
            self.adam_line.figure.canvas.draw()

        if self.amsgrad_model:
            self.amsgrad_line, = self.ax.plot(xs, self.amsgrad_loss, marker='v', label='AMSGrad optimizer')
            self.amsgrad_line.figure.canvas.draw()

        self.ax.legend(loc='upper right')

        self.canvas.draw()

    def calc_init_loss(self):
        init_loss_noopt, init_loss_rmsprop, init_loss_adam, init_loss_amsgrad = (
            run_init_loss(self.noopt_model,
                          self.rmsprop_model,
                          self.adam_model,
                          self.amsgrad_model,
                          self.data))

        self.noopt_loss.append(init_loss_noopt)
        self.rmsprop_loss.append(init_loss_rmsprop)
        self.adam_loss.append(init_loss_adam)
        self.amsgrad_loss.append(init_loss_amsgrad)

        self.epochs_done = self.epochs_done + 1

        self.update_plot()

    def make_epoch(self):
        training_loss_noopt, training_loss_rmsprop, training_loss_adam, training_loss_amsgrad = (
            run(self.noopt_model,
                self.rmsprop_model,
                self.adam_model,
                self.amsgrad_model,
                1,
                self.learning_rate,
                self.data))

        self.noopt_loss.append(training_loss_noopt)
        self.rmsprop_loss.append(training_loss_rmsprop)
        self.adam_loss.append(training_loss_adam)
        self.amsgrad_loss.append(training_loss_amsgrad)

        self.epochs_done = self.epochs_done + 1

        self.update_plot()

    def plot(self):
        self.epochs_done = 0

        self.noopt_loss = []
        self.rmsprop_loss = []
        self.adam_loss = []
        self.amsgrad_loss = []

        self.clear_plot()

        self.ax = self.canvas.figure.subplots()
        self.ax.set_title('Demonstration of backpropagation using optimizers')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')

        self.movie_label.setHidden(False)
        self.start_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

        if self.epochs_done == 0:
            self.init_models()
            self.calc_init_loss()

        for i in range(self.epochs_done, self.nb_epochs):
            self.make_epoch()

        self.update_plot()

        self.movie_label.setHidden(True)
        self.start_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

    def reset(self):
        self.nb_epochs_edit.setText("10")
        self.lr_edit.setText("0.01")

        self.noopt_chbox.setChecked(Qt.Checked)
        self.rmsprop_chbox.setChecked(Qt.Unchecked)
        self.adam_chbox.setChecked(Qt.Checked)
        self.amsgrad_chbox.setChecked(Qt.Unchecked)

        self.epochs_done = 0
        self.noopt_loss = []
        self.rmsprop_loss = []
        self.adam_loss = []
        self.amsgrad_loss = []

        self.clear_plot()

    def set_noopt(self):
        if not self.noopt_chbox.isChecked():
            if not (self.rmsprop_chbox.isChecked()
                    or self.adam_chbox.isChecked()
                    or self.amsgrad_chbox.isChecked()):
                self.noopt_chbox.setChecked(Qt.Checked)

    def set_rmsprop(self):
        if not self.rmsprop_chbox.isChecked():
            if not (self.noopt_chbox.isChecked()
                    or self.adam_chbox.isChecked()
                    or self.amsgrad_chbox.isChecked()):
                self.rmsprop_chbox.setChecked(Qt.Checked)

    def set_adam(self):
        if not self.adam_chbox.isChecked():
            if not (self.noopt_chbox.isChecked()
                    or self.rmsprop_chbox.isChecked()
                    or self.amsgrad_chbox.isChecked()):
                self.adam_chbox.setChecked(Qt.Checked)

    def set_amsgrad(self):
        if not self.amsgrad_chbox.isChecked():
            if not (self.noopt_chbox.isChecked()
                    or self.rmsprop_chbox.isChecked()
                    or self.adam_chbox.isChecked()):
                self.amsgrad_chbox.setChecked(Qt.Checked)

    def ui_setup(self, window):
        if not window.objectName():
            window.setObjectName(u"MainWindowUI")

        window.resize(self.window_width, self.window_height)
        window.setWindowTitle("MNIST demonstration of backpropagation using optimizers")

        ####################
        #      Widget      #
        ####################

        self.widget = QWidget(window)
        self.widget.setObjectName(u"widget")
        self.widget.setMinimumSize(self.min_window_width, self.min_window_height)

        self.horizontal_layout_widget = QWidget(self.widget)
        self.horizontal_layout_widget.setObjectName(u"horizontalLayoutWidget")
        self.horizontal_layout_widget.setGeometry(QRect(300, 10, 800, 400))

        ####################
        #     Header       #
        ####################

        self.hpparams_label = QLabel(self.widget)
        self.hpparams_label.setObjectName(u"hpparamsLabel")
        self.hpparams_label.setGeometry(QRect(10, 10, 165, 25))
        self.hpparams_label.setText("Hyperparameters")
        self.hpparams_label.setFont(self.header_font)

        self.opts_label = QLabel(self.widget)
        self.opts_label.setObjectName(u"optsLabel")
        self.opts_label.setGeometry(QRect(10, 170, 105, 25))
        self.opts_label.setText("Optimizers")
        self.opts_label.setFont(self.header_font)

        ####################
        # Number of epochs #
        ####################

        self.nb_epochs_label = QLabel(self.widget)
        self.nb_epochs_label.setObjectName(u"nbEpochsLabel")
        self.nb_epochs_label.setGeometry(QRect(10, 40, 125, 25))
        self.nb_epochs_label.setText("Number of epochs:")
        self.nb_epochs_label.setFont(self.label_font)

        self.nb_epochs_edit = QLineEdit(self.widget)
        self.nb_epochs_edit.setObjectName(u"nbEpochsEdit")
        self.nb_epochs_edit.setGeometry(QRect(10, 65, 32, 25))
        self.nb_epochs_edit.setToolTip("Number of epochs")
        self.nb_epochs_edit.setText("10")

        self.nb_epochs_validator = QIntValidator(1, 100)
        self.nb_epochs_validator.setLocale(QLocale("en_US"))
        self.nb_epochs_edit.setValidator(self.nb_epochs_validator)

        ####################
        #  Learning rate   #
        ####################

        self.lr_label = QLabel(self.widget)
        self.lr_label.setObjectName(u"lrLabel")
        self.lr_label.setGeometry(QRect(10, 95, 95, 25))
        self.lr_label.setText("Learning rate:")
        self.lr_label.setFont(self.label_font)

        self.lr_edit = QLineEdit(self.widget)
        self.lr_edit.setObjectName(u"lrEdit")
        self.lr_edit.setGeometry(QRect(10, 120, 60, 25))
        self.lr_edit.setToolTip("Learning rate")
        self.lr_edit.setText("0.01")

        self.lr_validator = QDoubleValidator(bottom=0.00001, top=1.0, decimals=5)
        self.lr_validator.setLocale(QLocale("en_US"))
        self.lr_edit.setValidator(self.lr_validator)

        ####################
        #    Checkbox      #
        ####################

        self.noopt_chbox = QCheckBox(self.widget)
        self.noopt_chbox.setObjectName(u"nooptChbox")
        self.noopt_chbox.setGeometry(QRect(10, 200, 110, 25))
        self.noopt_chbox.setChecked(Qt.Checked)
        self.noopt_chbox.toggled.connect(self.set_noopt)
        self.noopt_chbox.setText("No optimizer")
        self.noopt_chbox.setFont(self.label_font)

        self.rmsprop_chbox = QCheckBox(self.widget)
        self.rmsprop_chbox.setObjectName(u"rmspropChbox")
        self.rmsprop_chbox.setGeometry(QRect(10, 225, 155, 25))
        self.rmsprop_chbox.setChecked(Qt.Unchecked)
        self.rmsprop_chbox.toggled.connect(self.set_rmsprop)
        self.rmsprop_chbox.setText("RMSprop optimizer")
        self.rmsprop_chbox.setFont(self.label_font)

        self.adam_chbox = QCheckBox(self.widget)
        self.adam_chbox.setObjectName(u"adamChbox")
        self.adam_chbox.setGeometry(QRect(10, 250, 130, 25))
        self.adam_chbox.setChecked(Qt.Checked)
        self.adam_chbox.toggled.connect(self.set_adam)
        self.adam_chbox.setText("Adam optimizer")
        self.adam_chbox.setFont(self.label_font)

        self.amsgrad_chbox = QCheckBox(self.widget)
        self.amsgrad_chbox.setObjectName(u"amsgradChbox")
        self.amsgrad_chbox.setGeometry(QRect(10, 275, 155, 25))
        self.amsgrad_chbox.setChecked(Qt.Unchecked)
        self.amsgrad_chbox.toggled.connect(self.set_amsgrad)
        self.amsgrad_chbox.setText("AMSGrad optimizer")
        self.amsgrad_chbox.setFont(self.label_font)

        ####################
        #      Button      #
        ####################

        self.start_btn = QPushButton(self.widget)
        self.start_btn.setObjectName(u"startBtn")
        self.start_btn.setGeometry(QRect(10, 325, 60, 25))
        self.start_btn.clicked.connect(self.plot)
        self.start_btn.setText("Start")

        self.reset_btn = QPushButton(self.widget)
        self.reset_btn.setObjectName(u"resetBtn")
        self.reset_btn.setGeometry(QRect(90, 325, 71, 25))
        self.reset_btn.clicked.connect(self.reset)
        self.reset_btn.setText("Reset")

        ####################
        #      Layout      #
        ####################

        self.horizontal_layout = QHBoxLayout(self.horizontal_layout_widget)
        self.horizontal_layout.setObjectName(u"horizontalLayout")
        self.horizontal_layout.setContentsMargins(0, 0, 0, 0)

        ####################
        #      Canvas      #
        ####################

        self.canvas = FigureCanvas(Figure(figsize=(5, 3), tight_layout=True))

        ####################
        #      Movie       #
        ####################

        self.movie_label = QLabel(self.widget)
        self.movie_label.setObjectName(u"movieLabel")
        self.movie_label.setGeometry(QRect(700, 425, 50, 50))
        movie = QMovie('data/processing.gif')
        self.movie_label.setMovie(movie)
        self.movie_label.setHidden(True)
        movie.start()

        ####################

        self.clear_plot()

        window.setCentralWidget(self.widget)
        QMetaObject.connectSlotsByName(window)
