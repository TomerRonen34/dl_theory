from functools import partial

import numpy as np
import torch
from typing import *


def gaussian_init(shape: Tuple[int],
                  std: float = 1.,
                  mean: float = 0.
                  ) -> torch.Tensor:
    weights = torch.randn(shape,requires_grad=True) * std + mean
    return weights


def xavier_init(shape: Tuple[int, int]):
    fan_in, fan_out = shape
    magnitude = np.sqrt(6 / (fan_in + fan_out))
    weights = magnitude * (2 * torch.rand(shape,requires_grad=True) - 1)
    return weights


def get_init_func(init_type: str, init_gaussian_std: float):
    if init_type.lower() == "gaussian":
        if init_gaussian_std is None:
            raise ValueError("init_gaussian_std can't be None when using gaussian initialization")
        return partial(gaussian_init, std=init_gaussian_std, mean=0.)
    elif init_type.lower() == "xavier":
        return xavier_init
    else:
        raise ValueError('init_type should be one of ["gaussian", "xavier"]')