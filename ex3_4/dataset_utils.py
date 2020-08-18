from typing import List, Tuple, TypeVar, Dict, Iterable, Union
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

from utils import string_to_random_seed


def get_cifar_data_loaders(data_root: str,
                           classes_to_keep: Iterable[str] = None,
                           fraction_to_keep_train: float = 1.,
                           fraction_to_keep_test: float = 1.,
                           batch_size: int = 32,
                           num_loader_workers: int = 4,
                           random_train_targets: bool = False,
                           specific_adversarial_class_fractions: Dict[str, float] = None
                           ) -> Tuple[DataLoader, DataLoader]:
    trainloader = _get_cifar_data_loader(data_root=data_root,
                                         is_train=True,
                                         classes_to_keep=classes_to_keep,
                                         fraction_to_keep=fraction_to_keep_train,
                                         batch_size=batch_size,
                                         num_loader_workers=num_loader_workers,
                                         random_targets=random_train_targets,
                                         specific_adversarial_class_fractions=specific_adversarial_class_fractions)
    testloader = _get_cifar_data_loader(data_root=data_root,
                                        is_train=False,
                                        classes_to_keep=classes_to_keep,
                                        fraction_to_keep=fraction_to_keep_test,
                                        batch_size=batch_size,
                                        num_loader_workers=num_loader_workers,
                                        random_targets=False,
                                        specific_adversarial_class_fractions=None)
    return trainloader, testloader


def _get_cifar_data_loader(data_root: str,
                           is_train: bool,
                           classes_to_keep: Union[List[str], None],
                           fraction_to_keep: float,
                           batch_size: int,
                           num_loader_workers: int,
                           random_targets: bool,
                           specific_adversarial_class_fractions: Union[Dict[str, float], None]
                           ) -> DataLoader:
    transform = _get_image_transform()
    dataset = torchvision.datasets.CIFAR10(root=data_root, train=is_train,
                                           download=True, transform=transform)
    if classes_to_keep is not None:
        _keep_specific_classes_in_dataset(dataset, classes_to_keep)
    if fraction_to_keep != 1.:
        _keep_random_subset_of_dataset(dataset, fraction_to_keep)
    if random_targets:
        _randomize_targets_of_dataset(dataset)
    if specific_adversarial_class_fractions is not None:
        _randomize_specific_class_fractions(dataset, specific_adversarial_class_fractions)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_loader_workers)
    return dataloader


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


def _randomize_targets_of_dataset(dataset: Dataset,
                                  random_seed: int = 34
                                  ) -> None:
    """ Be warned - this function changes the dataset inplace """
    num_samples = len(dataset.targets)
    num_classes = len(dataset.classes)
    random_state = np.random.RandomState(random_seed)
    random_targets = random_state.randint(low=0, high=num_classes, size=num_samples).tolist()
    dataset.targets = random_targets


def _randomize_specific_class_fractions(dataset: Dataset,
                                        specific_adversarial_class_fractions: Dict[str, float],
                                        random_seed: int = 34):
    """ Be warned - this function changes the dataset inplace """
    num_classes = len(dataset.classes)
    new_targets = np.array(dataset.targets)
    for class_name, fraction_to_randomize in specific_adversarial_class_fractions.items():
        class_seed = random_seed + string_to_random_seed(class_name)
        random_state = np.random.RandomState(class_seed)

        class_target = dataset.classes.index(class_name)
        class_samples, = np.nonzero(np.array(dataset.targets) == class_target)
        num_class_samples = len(class_samples)

        num_randomize = int(num_class_samples * fraction_to_randomize)
        perm = random_state.permutation(num_class_samples)
        samples_to_randomize = class_samples[perm[:num_randomize]]

        random_targets = random_state.randint(low=0, high=num_classes, size=num_randomize)
        new_targets[samples_to_randomize] = random_targets

    dataset.targets = new_targets.tolist()
