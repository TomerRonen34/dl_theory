from typing import *
from nptyping import *

import os
import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


def fit_and_save(model, X_train, y_train,
                 save_dir, model_name):
    model.fit(X_train, y_train)
    save_model(model, save_dir, model_name)
    return model


def eval_and_plot(model, X_test, y_test,
                  class_names, save_dir, model_name, model_display_name):
    y_pred = model.predict(X_test)
    disp = plot_model_results(y_true=y_test, y_pred=y_pred,
                              class_names=class_names, model_display_name=model_display_name)
    save_confusion_matrix_display(disp, save_dir, model_name)


def plot_model_results(y_true, y_pred, class_names=None, model_display_name=None, figsize=(10, 10)):
    model_display_name = model_display_name if model_display_name is not None else ''
    accuracy = accuracy_score(y_true, y_pred)
    title = f"{model_display_name}\nAccuracy = {accuracy:.2f}"
    disp = plot_confusion_matrix(y_true, y_pred, class_names, figsize=figsize, title=title)
    plt.show()
    return disp


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 10), normalize="true", title=None):
    """
    very similar to sklearn.metrics.plot_confusion_metrix, but takes (y_true, y_pred) instead
    of (classifier, X, y_true)
    """
    labels = list(set(y_true).union(set(y_pred)))
    if class_names is None:
        class_names = list(map(str, labels))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation="vertical")
    prettify_confusion_matrix_display(disp, title)

    return disp


def prettify_confusion_matrix_display(disp: ConfusionMatrixDisplay,
                                      title: str):
    disp.ax_.set_title(title, {"fontsize": "x-large"})
    disp.ax_.tick_params(labelsize="large")
    disp.ax_.set_xlabel(disp.ax_.get_xlabel(), {"fontsize": "x-large"})
    disp.ax_.set_ylabel(disp.ax_.get_ylabel(), {"fontsize": "x-large"})


def save_confusion_matrix_display(disp: ConfusionMatrixDisplay,
                                  save_dir: str,
                                  model_name: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, model_name + ".png")
    disp.figure_.savefig(save_path, bbox_inches="tight")


def save_model(model, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, model_name + ".pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(save_dir, model_name):
    save_path = osp.join(save_dir, model_name + ".pkl")
    with open(save_path, 'rb') as f:
        model = pickle.load(f)
    return model


def shuffle_multiple_arrays(*arrays: NDArray, seed: int = None
                            ) -> Tuple[NDArray, ...]:
    length = len(arrays[0])
    random_state = np.random.RandomState(seed)
    perm = random_state.permutation(length)
    shuf_arrays = tuple([x[perm] for x in arrays])
    return shuf_arrays


def batchify(X: NDArray,
             y: NDArray,
             batch_size: int,
             seed: int = None
             ) -> List[Tuple[NDArray, NDArray]]:
    X, y = shuffle_multiple_arrays(X, y, seed=seed)
    num_batches = np.ceil(len(X) / batch_size)
    X_batches = np.array_split(X, num_batches)
    y_batches = np.array_split(y, num_batches)
    batches = list(zip(X_batches, y_batches))
    return batches
