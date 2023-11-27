# Project: Demonstration of backpropagation learning - basic algorithm and selected optimizer
# Author: David Chocholaty <xchoch09@stud.fit.vutbr.cz>
# File: train.py

from engine.backend.tensor import Tensor

from PyQt5.QtWidgets import QApplication

import numpy as np


def cross_entropy(scores, labels):
    return -((labels * scores.log()).sum())


# The following source inspired the source code of the following function.
# Source: https://github.com/hkxIron/tensorgrad/blob/6098d54eeeeeebf69ee89a2dcb0a7d8b60b95c16/test/test_network.py#L68
def loss(batch_size=None, X=None, y=None, model=None):
    # inline DataLoader
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]

    Xb = np.expand_dims(Xb, axis=0)
    inputs = Tensor(Xb)

    model_pred = model(inputs)
    scores = model_pred.reshape(-1)  # Model outputs.

    yb_tensor = Tensor(yb)  # Expected label one-hot.

    ###############################
    ###############################

    loss = cross_entropy(scores, yb_tensor)

    ####################
    # L2 regularization
    ####################
    alpha = 1e-4
    reg_loss = 0

    for p in model.parameters():
        reg_loss += (p * p).sum()
    ####################

    total_loss = loss + reg_loss * alpha

    score = scores.numpy()
    label = yb_tensor.numpy()

    accuracy = int(np.argmax(score) == np.argmax(label))

    return total_loss, accuracy


def get_init_loss(model, data):
    store_init_loss = []

    init_loss = 0.0

    for i in range(len(data.training_images)):
        if i % 100 == 0:
            QApplication.processEvents()

        total_loss, acc = loss(X=data.training_images[i], y=data.training_labels[i], model=model)

        init_loss += total_loss.data

    store_init_loss.append(init_loss / len(data.training_images))

    return store_init_loss


def train(model, nb_epochs, learning_rate, data):
    store_training_loss = []

    # Optimization
    for epoch in range(nb_epochs):
        training_loss = 0.0
        # training_acc = 0

        #################
        # Training step #
        #################
        for i in range(len(data.training_images)):
            if i % 100 == 0:
                QApplication.processEvents()

            # Forward pass
            total_loss, acc = loss(X=data.training_images[i], y=data.training_labels[i], model=model)

            training_loss += total_loss.data
            # training_acc += acc

            # Backward pass
            model.zero_grad()
            total_loss.backward()

            # Update (sgd)
            model.opt.update_params(model, learning_rate)

        store_training_loss.append(training_loss / len(data.training_images))

        # if epoch % 1 == 0: print( f"Epoch {epoch + 1}/{nb_epochs}: loss {training_loss / len(
        # data.training_images)}, accuracy {(training_acc / len(data.training_images)) * 100}%")

        return store_training_loss


def run_init_loss(noopt_model,
                  rmsprop_model,
                  adam_model,
                  amsgrad_model,
                  data):
    init_loss_noopt = []
    init_loss_rmsprop = []
    init_loss_adam = []
    init_loss_amsgrad = []

    if noopt_model:
        init_loss_noopt = get_init_loss(noopt_model, data)

    if rmsprop_model:
        init_loss_rmsprop = get_init_loss(rmsprop_model, data)

    if adam_model:
        init_loss_adam = get_init_loss(adam_model, data)

    if amsgrad_model:
        init_loss_amsgrad = get_init_loss(amsgrad_model, data)

    return (init_loss_noopt,
            init_loss_rmsprop,
            init_loss_adam,
            init_loss_amsgrad)


def run(noopt_model,
        rmsprop_model,
        adam_model,
        amsgrad_model,
        nb_epochs,
        learning_rate,
        data):
    training_loss_noopt = []
    training_loss_rmsprop = []
    training_loss_adam = []
    training_loss_amsgrad = []

    if noopt_model:
        training_loss_noopt = train(noopt_model, nb_epochs, learning_rate, data)

    if rmsprop_model:
        training_loss_rmsprop = train(rmsprop_model, nb_epochs, learning_rate, data)

    if adam_model:
        training_loss_adam = train(adam_model, nb_epochs, learning_rate, data)

    if amsgrad_model:
        training_loss_amsgrad = train(amsgrad_model, nb_epochs, learning_rate, data)

    return (training_loss_noopt,
            training_loss_rmsprop,
            training_loss_adam,
            training_loss_amsgrad)
