from typing import List, Tuple, TypeVar
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


def get_cifar_data_loaders(data_root: str,
                           classes_to_keep: List[str] = None,
                           fraction_to_keep_train: float = 1.,
                           fraction_to_keep_test: float = 1.,
                           batch_size: int = 32,
                           num_loader_workers: int = 4
                           ) -> Tuple[DataLoader, DataLoader]:
    trainloader = _get_cifar_data_loader(data_root=data_root,
                                         is_train=True,
                                         classes_to_keep=classes_to_keep,
                                         fraction_to_keep=fraction_to_keep_train,
                                         batch_size=batch_size,
                                         num_loader_workers=num_loader_workers)
    testloader = _get_cifar_data_loader(data_root=data_root,
                                        is_train=False,
                                        classes_to_keep=classes_to_keep,
                                        fraction_to_keep=fraction_to_keep_test,
                                        batch_size=batch_size,
                                        num_loader_workers=num_loader_workers)
    return trainloader, testloader


def _get_cifar_data_loader(data_root: str,
                           is_train: bool,
                           classes_to_keep: List[str],
                           fraction_to_keep: float,
                           batch_size: int,
                           num_loader_workers: int
                           ) -> DataLoader:
    transform = _get_image_transform()
    dataset = torchvision.datasets.CIFAR10(root=data_root, train=is_train,
                                           download=True, transform=transform)
    if classes_to_keep is not None:
        _keep_specific_classes_in_dataset(dataset, classes_to_keep)
    if fraction_to_keep != 1.:
        _keep_random_subset_of_dataset(dataset, fraction_to_keep)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_loader_workers)
    return dataloader


def _keep_specific_classes_in_dataset(dataset: Dataset,
                                      classes_to_keep: List[str]
                                      ) -> None:
    """ Be warned - this function changes the dataset inplace """
    target_classes = np.array(dataset.classes)[dataset.targets]
    is_keep = pd.Series(target_classes).isin(classes_to_keep).values

    target_classes = target_classes[is_keep]
    classes, targets = np.unique(target_classes, return_inverse=True)

    dataset.data = dataset.data[is_keep]
    dataset.targets = targets.tolist()
    dataset.classes = classes.tolist()


def _keep_random_subset_of_dataset(dataset: Dataset,
                                   fraction_to_keep: float = 0.1,
                                   random_seed: int = 34
                                   ) -> None:
    """ Be warned - this function changes the dataset inplace """
    num_samples = len(dataset.data)
    num_keep = int(fraction_to_keep * num_samples)

    random_state = np.random.RandomState(seed=random_seed)
    samples_to_keep = random_state.permutation(num_samples)[:num_keep]

    dataset.data = dataset.data[samples_to_keep]
    dataset.targets = np.array(dataset.targets)[samples_to_keep].tolist()


Transform = TypeVar("Transform")


def _get_image_transform() -> Transform:
    """
    The output of torchvision datasets are PILImage images of range [0, 1].
    We transform them to Tensors of normalized range [-1, 1].
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform
