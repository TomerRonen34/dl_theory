from typing import *
from nptyping import *
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


def shuffle_multiple_arrays(*arrays: NDArray, seed: int = None
                            ) -> Tuple[NDArray, ...]:
    length = len(arrays[0])
    random_state = np.random.RandomState(seed)
    perm = random_state.permutation(length)
    shuf_arrays = tuple([x[perm] for x in arrays])
    return shuf_arrays


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
    disp.ax_.set_title(title)
    plt.show()


def plot_model_results(y_true, y_pred, class_names=None, model_name=None, figsize=(10, 10)):
    model_name = model_name if model_name is not None else ''
    accuracy = accuracy_score(y_true, y_pred)
    title = f"{model_name}\nAccuracy = {accuracy:.2f}"
    plot_confusion_matrix(y_true, y_pred, class_names, figsize=figsize, title=title)
