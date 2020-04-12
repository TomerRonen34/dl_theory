import torch
from typing import *


def gaussian_init(shape: Tuple[int],
                  std: float = 1.,
                  mean: float = 0.
                  ) -> torch.Tensor:
    return torch.randn(shape) * std + mean
