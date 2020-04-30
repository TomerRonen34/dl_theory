import numpy as np
import pickle
import os
import os.path as osp
from utils import shuffle_multiple_arrays, device
from typing import *
from nptyping import *
import torch

def load_cifar_dataset(dataset_dir: str
                       ) -> Tuple[NDArray[(Any, 32, 32, 3), np.uint8],
                                  NDArray[Any, int],
                                  NDArray[(Any, 32, 32, 3), np.uint8],
                                  NDArray[Any, int],
                                  NDArray[Any, str]]:
    train_paths, test_path, meta_path = _generate_paths(dataset_dir)
    train_images, train_labels = _load_multiple_batches(train_paths)
    test_images, test_labels = _load_batch(test_path)
    class_names = _load_class_names(meta_path)
    return train_images, train_labels, test_images, test_labels, class_names


def prepare_cifar_data_for_classifier(dataset_dir: str, cache_dir: str, subsample_fraction: float = 0.1,
                                      seed: int = 34, keep_as_image=False):
    if cache_dir is None:
        return _prepare_for_classifier(dataset_dir, keep_as_image, subsample_fraction, seed)

    os.makedirs(cache_dir, exist_ok=True)

    if keep_as_image:
        dim_str = 'as_image'
    else:
        dim_str = 'as_vector'

    if subsample_fraction is None:
        data_dump_path = osp.join(
            cache_dir, f"cifar_vectors_seed_{seed}.npz")
    else:
        data_dump_path = osp.join(
            cache_dir, f"cifar_vectors_subsample_{subsample_fraction:.2f}_seed_{seed}_{dim_str}.npz")

    if not osp.exists(data_dump_path):
        X_train, y_train, X_test, y_test, class_names = (
            _prepare_for_classifier(dataset_dir, keep_as_image, subsample_fraction, seed))
        np.savez(data_dump_path, X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test, class_names=class_names)
    else:
        npzfile = np.load(data_dump_path)
        X_train, y_train, X_test, y_test, class_names = (
            npzfile["X_train"], npzfile["y_train"],
            npzfile["X_test"], npzfile["y_test"], npzfile["class_names"])

    return X_train, y_train, X_test, y_test, class_names


def _prepare_for_classifier(dataset_dir: str, keep_as_image=False, subsample_fraction: float = None,
                            seed: int = 34):
    orig_train_images, orig_train_labels, orig_test_images, orig_test_labels, class_names = (
        load_cifar_dataset(dataset_dir))
    train_images, y_train = shuffle_multiple_arrays(orig_train_images, orig_train_labels, seed=seed)
    test_images, y_test = shuffle_multiple_arrays(orig_test_images, orig_test_labels, seed=seed)

    if keep_as_image:
        X_train = np.rollaxis(train_images, 3, 1) / 255.
        X_test = np.rollaxis(test_images, 3, 1) / 255.
    else:
        X_train = train_images.reshape(len(train_images), -1) / 255.
        X_test = test_images.reshape(len(test_images), -1) / 255.

    if subsample_fraction is not None:
        num_train = int(len(X_train) * subsample_fraction)
        num_test = int(len(X_test) * subsample_fraction)
        X_train, y_train, X_test, y_test = (
            X_train[:num_train], y_train[:num_train], X_test[:num_test], y_test[:num_test])

    return X_train, y_train, X_test, y_test, class_names


def _generate_paths(dataset_dir: str
                    ) -> Tuple[List[str],
                               str,
                               str]:
    train_paths = [osp.join(dataset_dir, f"data_batch_{i}") for i in range(1, 6)]
    test_path = osp.join(dataset_dir, "test_batch")
    meta_path = osp.join(dataset_dir, "batches.meta")
    return train_paths, test_path, meta_path


def _load_multiple_batches(batch_paths: Iterable[str]
                           ) -> Tuple[NDArray[(Any, 32, 32, 3), np.uint8],
                                      NDArray[Any, int]]:
    loaded_batches = [_load_batch(path) for path in batch_paths]
    images, labels = zip(*loaded_batches)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels


def _load_batch(batch_path: str
                ) -> Tuple[NDArray[(Any, 32, 32, 3), np.uint8],
                           NDArray[Any, int]]:
    batch_dict = _load_cifar_file(batch_path)
    labels = np.asarray(batch_dict[b"labels"])
    image_vectors = np.asarray(batch_dict[b"data"])
    images = _unpack_image_vectors(image_vectors)
    return images, labels


def _load_cifar_file(path: str
                     ) -> Dict[ByteString, Any]:
    with open(path, 'rb') as f:
        cifar_dict = pickle.load(f, encoding='bytes')
    return cifar_dict


def _unpack_image_vectors(image_vectors: NDArray[(Any, 32 * 32 * 3), np.uint8]
                          ) -> NDArray[(Any, 32, 32, 3), np.uint8]:
    images = np.reshape(image_vectors, (-1, 3, 32, 32)
                        ).transpose((0, 2, 3, 1))
    return images


def _load_class_names(meta_path: str
                      ) -> NDArray[Any, str]:
    cifar_meta_dict = _load_cifar_file(meta_path)
    bytes_class_names = cifar_meta_dict[b"label_names"]
    class_names = np.array([cls.decode() for cls in bytes_class_names])
    return class_names


def _example():
    train_images, train_labels, test_images, test_labels, class_names = (
        load_cifar_dataset("cifar-10-batches-py"))

    import matplotlib.pyplot as plt
    def show_image(image, title):
        plt.figure()
        plt.imshow(image)
        plt.title(title)
        plt.show()

    show_image(train_images[0], class_names[train_labels[0]])
    show_image(test_images[0], class_names[test_labels[0]])
    pass


if __name__ == '__main__':
    _example()
