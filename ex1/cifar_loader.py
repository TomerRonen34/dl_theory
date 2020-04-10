import numpy as np
import pickle
import os.path as osp
from typing import *
from nptyping import *


def load_cifar_dataset(dataset_dir):
    train_paths, test_path, meta_path = _generate_paths(dataset_dir)
    train_images, train_labels = _load_multiple_batches(train_paths)
    test_images, test_labels = _load_batch(test_path)
    class_names = _load_class_names(meta_path)
    return train_images, train_labels, test_images, test_labels, class_names


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
                                      NDArray[int]]:
    loaded_batches = [_load_batch(path) for path in batch_paths]
    images, labels = zip(*loaded_batches)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels


def _load_batch(batch_path: str
                ) -> Tuple[NDArray[(Any, 32, 32, 3), np.uint8],
                           NDArray[int]]:
    batch_dict = _load_cifar_file(batch_path)
    labels = np.asarray(batch_dict[b"labels"])
    image_vectors = np.asarray(batch_dict[b"data"])
    images = _parse_image_vectors(image_vectors)
    return images, labels


def _load_cifar_file(path: str
                     ) -> Dict[ByteString, Any]:
    with open(path, 'rb') as f:
        cifar_dict = pickle.load(f, encoding='bytes')
    return cifar_dict


def _parse_image_vectors(image_vectors: NDArray[(Any, 32 * 32 * 3), np.uint8]
                         ) -> NDArray[(Any, 32, 32, 3), np.uint8]:
    # TODO: numpy-ize
    images = []
    for _image_vector in image_vectors:
        _image = np.reshape(_image_vector, (3, 32, 32)).transpose((1, 2, 0))
        images.append(_image)
    images = np.array(images)
    return images


def _load_class_names(meta_path: str
                      ) -> List[str]:
    cifar_meta_dict = _load_cifar_file(meta_path)
    bytes_class_names = cifar_meta_dict[b"label_names"]
    class_names = [cls.decode() for cls in bytes_class_names]
    return class_names


def _example():
    train_images, train_labels = load_cifar_dataset("cifar-10-batches-py")
    pass


if __name__ == '__main__':
    _example()
