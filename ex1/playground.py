#!/usr/bin/env python
# coding: utf-8

import os.path as osp
import numpy as np
import torch
from fully_connected import FullyConnectedClassifier

from cifar_loader import load_cifar_dataset
from utils import shuffle_multiple_arrays
from training import fit_classifier

seed = 34
save_dir = "models"

# ### Prepare data

data_dump_path = "data.npz"
if not osp.exists(data_dump_path):
    orig_train_images, orig_train_labels, orig_test_images, orig_test_labels, class_names = (
        load_cifar_dataset("cifar-10-batches-py"))
    train_images, train_labels = shuffle_multiple_arrays(orig_train_images, orig_train_labels, seed=seed)
    test_images, test_labels = shuffle_multiple_arrays(orig_test_images, orig_test_labels, seed=seed)

    X_train = train_images.reshape(len(train_images), -1)
    X_test = test_images.reshape(len(test_images), -1)
    y_train = train_labels
    y_test = test_labels

    subsample_fraction = 0.1
    num_train = int(len(X_train) * subsample_fraction)
    num_test = int(len(X_test) * subsample_fraction)
    X_train, y_train, X_test, y_test = (
        X_train[:num_train], y_train[:num_train], X_test[:num_test], y_test[:num_test])
    print(f"num_train: {num_train}   num_test: {num_test}")

    np.savez(data_dump_path, X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test, class_names=class_names)
else:
    npzfile = np.load(data_dump_path)
    X_train, y_train, X_test, y_test, class_names = (
        npzfile["X_train"], npzfile["y_train"],
        npzfile["X_test"], npzfile["y_test"], npzfile["class_names"])

# ## Training a fully connected net

hidden_size = 256
num_hidden_layers = 1
activation = "relu"
init_type = "gaussian"
init_gaussian_std = 0.001
epochs = 50
batch_size = 32
learning_rate = 0.0001

net = FullyConnectedClassifier(num_classes=len(class_names),
                               input_size=X_train.shape[1],
                               hidden_size=hidden_size,
                               num_hidden_layers=num_hidden_layers,
                               activation=activation,
                               init_type=init_type,
                               init_gaussian_std=init_gaussian_std)

optimizer = torch.optim.SGD(net.trainable_params(), lr=learning_rate)

fit_classifier(net, optimizer, X_train, y_train, epochs, batch_size, seed,
               report_progress=True,
               X_test=X_test,
               y_test=y_test,
               print_gradient_amplitudes=False)
