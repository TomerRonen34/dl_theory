import numpy as np
import torch
from typing import *


def gaussian_init(shape: Tuple[int],
                  std: float = 1.,
                  mean: float = 0.
                  ) -> torch.Tensor:
    weights = torch.randn(shape) * std + mean
    return weights


def xavier_init(shape: Tuple[int, int]):
    fan_in, fan_out = shape
    magnitude = np.sqrt(6 / (fan_in + fan_out))
    weights = magnitude * (2 * torch.rand(shape) - 1)
    return weights
