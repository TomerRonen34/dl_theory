from typing import List, Tuple, TypeVar, Dict, Iterable, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

from utils import string_to_random_seed


class AdversarialTargetReplacementParams:
    def __init__(self,
                 replace_from: List[str],
                 replace_to: List[str],
                 fraction_to_replace: List[float]):
        self.replace_from = replace_from
        self.replace_to = replace_to
        self.fraction_to_replace = fraction_to_replace


def get_cifar_data_loaders(data_root: str,
                           classes_to_keep: Iterable[str] = None,
                           fraction_to_keep_train: float = 1.,
                           fraction_to_keep_test: float = 1.,
                           batch_size: int = 32,
                           num_loader_workers: int = 4,
                           random_train_targets_fraction: float = 0.,
                           adversarial_target_replacement_params: AdversarialTargetReplacementParams = None
                           ) -> Tuple[DataLoader, DataLoader]:
    trainloader = _get_cifar_data_loader(data_root=data_root,
                                         is_train=True,
                                         classes_to_keep=classes_to_keep,
                                         fraction_to_keep=fraction_to_keep_train,
                                         batch_size=batch_size,
                                         num_loader_workers=num_loader_workers,
                                         random_targets_fraction=random_train_targets_fraction,
                                         adversarial_target_replacement_params=adversarial_target_replacement_params)
    testloader = _get_cifar_data_loader(data_root=data_root,
                                        is_train=False,
                                        classes_to_keep=classes_to_keep,
                                        fraction_to_keep=fraction_to_keep_test,
                                        batch_size=batch_size,
                                        num_loader_workers=num_loader_workers,
                                        random_targets_fraction=False,
                                        adversarial_target_replacement_params=None)
    return trainloader, testloader


def _get_cifar_data_loader(data_root: str,
                           is_train: bool,
                           classes_to_keep: Union[List[str], None],
                           fraction_to_keep: float,
                           batch_size: int,
                           num_loader_workers: int,
                           random_targets_fraction: float,
                           adversarial_target_replacement_params: Union[AdversarialTargetReplacementParams, None]
                           ) -> DataLoader:
    transform = _get_image_transform()
    dataset = torchvision.datasets.CIFAR10(root=data_root, train=is_train,
                                           download=True, transform=transform)
    if classes_to_keep is not None:
        _keep_specific_classes_in_dataset(dataset, classes_to_keep)
    if fraction_to_keep != 1.:
        _keep_random_subset_of_dataset(dataset, fraction_to_keep)
    if random_targets_fraction != 0.:
        _randomize_targets_of_dataset(dataset, random_targets_fraction)
    if adversarial_target_replacement_params is not None:
        _adversarially_replace_targets(dataset, adversarial_target_replacement_params)

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
                                  random_targets_fraction: float,
                                  random_seed: int = 34
                                  ) -> None:
    """ Be warned - this function changes the dataset inplace """
    num_samples = len(dataset.targets)
    num_classes = len(dataset.classes)
    num_randomize = int(num_samples * random_targets_fraction)
    random_state = np.random.RandomState(random_seed)

    new_targets = np.array(dataset.targets)
    samples_to_randomize = random_state.permutation(num_samples)[:num_randomize]
    random_targets = random_state.randint(low=0, high=num_classes, size=num_randomize)
    new_targets[samples_to_randomize] = random_targets

    dataset.targets = new_targets.tolist()


def _adversarially_replace_targets(dataset: Dataset,
                                   replacement_params: AdversarialTargetReplacementParams,
                                   random_seed: int = 34):
    """ Be warned - this function changes the dataset inplace """
    new_targets = np.array(dataset.targets)

    for class_from, class_to, fraction_to_replace in zip(
            replacement_params.replace_from,
            replacement_params.replace_to,
            replacement_params.fraction_to_replace):
        class_seed = random_seed + string_to_random_seed(class_from)
        random_state = np.random.RandomState(class_seed)

        target_from = dataset.classes.index(class_from)
        target_to = dataset.classes.index(class_to)
        class_samples, = np.nonzero(np.array(dataset.targets) == target_from)
        num_class_samples = len(class_samples)

        num_replace = int(num_class_samples * fraction_to_replace)
        perm = random_state.permutation(num_class_samples)
        samples_to_replace = class_samples[perm[:num_replace]]

        new_targets[samples_to_replace] = target_to

    dataset.targets = new_targets.tolist()
