import numpy as np
import pickle
import os.path as osp
from typing import *
from nptyping import *


def load_cifar_dataset(dataset_dir):
    train_paths, test_path, class_names_path = _generate_paths(dataset_dir)
    train_images, train_labels = _load_multiple_batches(train_paths)
    return train_images, train_labels


def _generate_paths(dataset_dir: str
                    ) -> Tuple[List[str], str, str]:
    train_paths = [osp.join(dataset_dir, f"data_batch_{i}") for i in range(1, 6)]
    test_path = osp.join(dataset_dir, "test_batch")
    class_names_path = "batches.meta"
    return train_paths, test_path, class_names_path


def _load_multiple_batches(batch_paths: Iterable[str]
                           ) -> Tuple[NDArray[(Any, 32, 32)], NDArray]:
    images = []
    labels = []
    for _batch_path in batch_paths:
        _images, _labels = _load_batch(_batch_path)
        images.append(_images)
        labels.append(_labels)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels


def _load_batch(batch_path: str
                ):  # -> Tuple[np.ndarray, np.ndarray]:
    batch_dict = _read_cifar_file(batch_path)
    labels = np.array(batch_dict[b"labels"])
    image_vectors = np.array(batch_dict[b"data"])
    images = _parse_image_vectors(image_vectors)
    return images, labels


def _read_cifar_file(path: str
                     ) -> Dict[ByteString, Any]:
    with open(path, 'rb') as f:
        cifar_dict = pickle.load(f, encoding='bytes')
    return cifar_dict


def _parse_image_vectors(image_vectors: np.ndarray
                         ) -> np.ndarray:
    images = []
    for _image_vector in image_vectors:
        _image = np.reshape(_image_vector, (3, 32, 32)).transpose((1, 2, 0))
        images.append(_image)
    images = np.array(images)
    return images


def _example():
    train_images, train_labels = load_cifar_dataset("cifar-10-batches-py")
    pass


if __name__ == '__main__':
    _example()
