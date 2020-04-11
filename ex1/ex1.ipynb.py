#!/usr/bin/env python
# coding: utf-8

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path as osp

import torch

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from cifar_loader import load_cifar_dataset
from utils import shuffle_multiple_arrays, plot_model_results


seed = 34


orig_train_images, orig_train_labels, test_images, test_labels, class_names = load_cifar_dataset("cifar-10-batches-py")
train_images, train_labels = shuffle_multiple_arrays(orig_train_images, orig_train_labels, seed=seed)


X_train = train_images.reshape(len(train_images), -1)
X_test = test_images.reshape(len(test_images), -1)


model = SVC(kernel="linear", random_state=seed)
model.fit(X_train[:1000], train_labels[:1000])


# y_pred = model.predict(X_test)
# y_true = test_labels
y_pred = model.predict(X_test[::10])
y_true = test_labels[::10]


plot_model_results(y_true, y_pred, class_names, model_name="Linear SVM")

