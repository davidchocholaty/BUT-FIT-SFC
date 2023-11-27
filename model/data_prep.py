# Project: Demonstration of backpropagation learning - basic algorithm and selected optimizer
# Author: David Chocholaty <xchoch09@stud.fit.vutbr.cz>
# File: data_prep.py

import sys

import numpy as np


# The function is taken over from the following source:
# Source: https://github.com/numpy/numpy-tutorials/blob/main/content/tutorial-deep-learning-on-mnist.md
def one_hot_encoding(labels, dimension=10):
    # Define a one-hot variable for an all-zero vector
    # with 10 dimensions (number labels from 0 to 9).
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    # Return one-hot encoded labels.
    return one_hot_labels.astype(np.float64)


class DataPreprocessor:
    def __init__(self):
        self.training_labels = None
        self.training_images = None
        self.x_train = None
        self.y_train = None
        self.train_data = None
        self.data = None
        # self.test_labels = None
        # self.test_images = None
        # self.x_test = None
        # self.y_test = None
        # self.test_data = None

    def create_dataset(self):
        try:
            self.data = np.genfromtxt('data/train_small.csv', delimiter=',', skip_header=1)
        except FileNotFoundError:
            print("\nError: the file \"data/train_small.csv\" does not exist. Please add the dataset file.")
            sys.exit(1)

        m, n = self.data.shape
        np.random.shuffle(self.data)

        # self.test_data = self.data[0:1000]
        # self.y_test = self.test_data[:, 0]
        # self.x_test = self.test_data[:, 1:n]

        self.train_data = self.data[1000:m]
        self.y_train = self.train_data[:, 0]
        self.x_train = self.train_data[:, 1:n]

        # training_sample, test_sample = 1000, 1000
        training_sample = 1000
        self.training_images = self.x_train[0:training_sample] / 255
        # self.test_images = self.x_test[0:test_sample] / 255

        self.training_labels = one_hot_encoding(self.y_train[:training_sample])
        # self.test_labels = one_hot_encoding(self.y_test[:test_sample])
