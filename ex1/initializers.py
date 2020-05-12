from functools import partial
from torch.autograd import Variable
import numpy as np
import torch
from typing import *
from utils import device


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def gaussian_init(shape: Tuple[int],
                  std: float = 1.,
                  mean: float = 0.
                  ) -> torch.Tensor:
    weights = torch.randn(shape, device=device) * std + mean
    return Variable(weights, requires_grad=True)


def xavier_init(shape: Tuple[int, int]):
    fan_in, fan_out = get_fans(shape)
    magnitude = np.sqrt(6 / (fan_in + fan_out))
    weights = magnitude * (2 * torch.rand(shape, device=device) - 1)
    return Variable(weights, requires_grad=True)


def zero_init(shape):
    return torch.zeros(shape, device=device, requires_grad=True)


def get_init_func(init_type: str, init_gaussian_std: float):
    if init_type.lower() == "gaussian":
        if init_gaussian_std is None:
            raise ValueError("init_gaussian_std can't be None when using gaussian initialization")
        return partial(gaussian_init, std=init_gaussian_std, mean=0.)
    elif init_type.lower() == "xavier":
        return xavier_init
    else:
        raise ValueError('init_type should be one of ["gaussian", "xavier"]')
