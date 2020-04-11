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

from cifar_loader import load_cifar_dataset
from utils import shuffle_multiple_arrays, fit_and_save, eval_and_plot


seed = 34
save_dir = "models"


# ### Prepare data

orig_train_images, orig_train_labels, test_images, test_labels, class_names = load_cifar_dataset("cifar-10-batches-py")
train_images, train_labels = shuffle_multiple_arrays(orig_train_images, orig_train_labels, seed=seed)

X_train = train_images.reshape(len(train_images), -1)
X_test = test_images.reshape(len(test_images), -1)
y_train = train_labels.copy()
y_test = test_labels.copy()


n = 20
X_train, y_train, X_test, y_test = X_train[:n], y_train[:n], X_test[:n], y_test[:n]


# ## Linear SVM

model = SVC(kernel="linear", random_state=seed)
model_name = "linear_svm"
model_display_name = "Linear SVM"

fit_and_save(model, X_train, y_train,
             save_dir, model_name)

eval_and_plot(model, X_test, y_test,
              class_names, save_dir, model_name, model_display_name)


# ## SVM with RBF Kernel

model = SVC(kernel="linear", random_state=seed)
model_name = "linear_svm"
model_display_name = "Linear SVM"

fit_and_save(model, X_train, y_train,
             save_dir, model_name)

eval_and_plot(model, X_test, y_test,
              class_names, save_dir, model_name, model_display_name)


model = SVC(kernel="rbf", random_state=seed)
model_name = "rbf_svm"
model_display_name = "SVM with RBF Kernel"

fit_and_save(model, X_train, y_train,
             save_dir, model_name)

eval_and_plot(model, X_test, y_test,
              class_names, save_dir, model_name, model_display_name)




